"""Prior-auth agent orchestration loop.

This module contains the multi-step tool-use workflow that:
1. Calls fetch_coverage_rule for the member/procedure
2. Parses and validates the structured tool result
3. Decides whether a follow-up policy document fetch is needed
4. Calls the LLM to synthesize a final PriorAuthSummary
5. Validates the LLM output before returning
"""

from __future__ import annotations

from dataclasses import asdict
import json
import uuid

from agent import llm_client, tools
from agent.config import config
from agent.schemas import (
    AgentResponse,
    CoverageRuleResult,
    PolicyDocumentResult,
    PriorAuthSummary,
    StructuredError,
    ToolResult,
    COVERAGE_RULE_REQUIRED_FIELDS,
    POLICY_DOCUMENT_REQUIRED_FIELDS,
    PRIOR_AUTH_SUMMARY_REQUIRED_FIELDS,
)

SYSTEM_PROMPT = """You are a healthcare benefits prior-authorization assistant.
Given coverage rule and policy document information, produce a structured summary.
Return a JSON object with exactly these fields:
- approval_required: boolean
- reason: string explaining the determination
- required_documents: array of strings listing required document types
"""


def _missing_fields(raw: dict, required_fields: set[str]) -> set[str]:
    """Return required fields that are absent from a raw object."""
    return {field for field in required_fields if field not in raw}


def _schema_error(code: str, message: str, details: dict | None = None) -> StructuredError:
    """Create a consistently-shaped schema/tool validation error."""
    return StructuredError(code=code, message=message, details=details or {})


def _validate_string(value: object, field_name: str, tool_name: str) -> StructuredError | None:
    if not isinstance(value, str):
        return _schema_error(
            "SCHEMA_INVALID_TYPE",
            f"Field '{field_name}' from {tool_name} must be a string.",
            {"field": field_name, "expected": "string", "actual": type(value).__name__},
        )
    return None


def _validate_bool(value: object, field_name: str, source: str) -> StructuredError | None:
    # Use exact type checking because bool is a subclass of int in Python.
    if type(value) is not bool:
        return _schema_error(
            "SCHEMA_INVALID_TYPE",
            f"Field '{field_name}' from {source} must be a boolean.",
            {"field": field_name, "expected": "boolean", "actual": type(value).__name__},
        )
    return None


def _validate_string_list(value: object, field_name: str, source: str) -> StructuredError | None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return _schema_error(
            "SCHEMA_INVALID_TYPE",
            f"Field '{field_name}' from {source} must be an array of strings.",
            {"field": field_name, "expected": "array[string]", "actual": type(value).__name__},
        )
    return None


def _validate_optional_number(value: object, field_name: str, source: str) -> StructuredError | None:
    if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        return _schema_error(
            "SCHEMA_INVALID_TYPE",
            f"Field '{field_name}' from {source} must be a number or null.",
            {"field": field_name, "expected": "number|null", "actual": type(value).__name__},
        )
    return None


def parse_tool_result(tool_name: str, raw_result: dict) -> ToolResult | StructuredError:
    """Parse and validate a tool's raw JSON output into a typed result.

    If required fields are missing, field types are invalid, the tool returned an
    ``error_code``, or the raw tool payload is malformed, return a
    ``StructuredError`` instead of raising. This prevents malformed local or LLM
    tool data from surfacing as unhandled dataclass construction errors.
    """
    if tool_name not in {"fetch_coverage_rule", "fetch_policy_document"}:
        return _schema_error(
            "UNKNOWN_TOOL",
            f"Unknown tool result cannot be parsed: {tool_name}",
            {"tool": tool_name},
        )

    if not isinstance(raw_result, dict):
        return _schema_error(
            "TOOL_MALFORMED_OUTPUT",
            f"Tool {tool_name} returned a non-object result.",
            {"tool": tool_name, "actual": type(raw_result).__name__},
        )

    if "error_code" in raw_result:
        return StructuredError(
            code=f"TOOL_{raw_result.get('error_code', 'ERROR')}",
            message=str(raw_result.get("message") or f"Tool {tool_name} returned an error."),
            details={"tool": tool_name, "raw_result": raw_result},
        )

    if tool_name == "fetch_coverage_rule":
        missing = _missing_fields(raw_result, COVERAGE_RULE_REQUIRED_FIELDS)
        if missing:
            return _schema_error(
                "SCHEMA_MISSING_FIELD",
                f"Coverage rule result is missing required field(s): {', '.join(sorted(missing))}",
                {"tool": tool_name, "missing_fields": sorted(missing), "raw_result": raw_result},
            )

        for field_name in ("member_id", "cpt_code", "plan_name", "reason"):
            err = _validate_string(raw_result.get(field_name), field_name, tool_name)
            if err:
                return err

        err = _validate_bool(raw_result.get("approval_required"), "approval_required", tool_name)
        if err:
            return err

        policy_document_id = raw_result.get("policy_document_id")
        if policy_document_id is not None and not isinstance(policy_document_id, str):
            return _schema_error(
                "SCHEMA_INVALID_TYPE",
                "Field 'policy_document_id' from fetch_coverage_rule must be a string or null.",
                {"field": "policy_document_id", "expected": "string|null", "actual": type(policy_document_id).__name__},
            )

        required_documents_hint = raw_result.get("required_documents_hint", [])
        err = _validate_string_list(required_documents_hint, "required_documents_hint", tool_name)
        if err:
            return err

        copay = raw_result.get("copay")
        err = _validate_optional_number(copay, "copay", tool_name)
        if err:
            return err

        deductible_remaining = raw_result.get("deductible_remaining")
        err = _validate_optional_number(deductible_remaining, "deductible_remaining", tool_name)
        if err:
            return err

        return CoverageRuleResult(
            member_id=raw_result["member_id"],
            cpt_code=raw_result["cpt_code"],
            plan_name=raw_result["plan_name"],
            approval_required=raw_result["approval_required"],
            reason=raw_result["reason"],
            policy_document_id=policy_document_id,
            required_documents_hint=list(required_documents_hint),
            copay=float(copay) if copay is not None else None,
            deductible_remaining=float(deductible_remaining) if deductible_remaining is not None else None,
        )

    missing = _missing_fields(raw_result, POLICY_DOCUMENT_REQUIRED_FIELDS)
    if missing:
        return _schema_error(
            "SCHEMA_MISSING_FIELD",
            f"Policy document result is missing required field(s): {', '.join(sorted(missing))}",
            {"tool": tool_name, "missing_fields": sorted(missing), "raw_result": raw_result},
        )

    for field_name in ("document_id", "title", "summary"):
        err = _validate_string(raw_result.get(field_name), field_name, tool_name)
        if err:
            return err

    err = _validate_string_list(raw_result.get("required_documents"), "required_documents", tool_name)
    if err:
        return err

    turnaround_days = raw_result.get("turnaround_days", 0)
    if not isinstance(turnaround_days, int) or isinstance(turnaround_days, bool):
        return _schema_error(
            "SCHEMA_INVALID_TYPE",
            "Field 'turnaround_days' from fetch_policy_document must be an integer when present.",
            {"field": "turnaround_days", "expected": "integer", "actual": type(turnaround_days).__name__},
        )

    submission_method = raw_result.get("submission_method", "")
    if not isinstance(submission_method, str):
        return _schema_error(
            "SCHEMA_INVALID_TYPE",
            "Field 'submission_method' from fetch_policy_document must be a string when present.",
            {"field": "submission_method", "expected": "string", "actual": type(submission_method).__name__},
        )

    return PolicyDocumentResult(
        document_id=raw_result["document_id"],
        title=raw_result["title"],
        summary=raw_result["summary"],
        required_documents=list(raw_result["required_documents"]),
        turnaround_days=turnaround_days,
        submission_method=submission_method,
    )


def should_call_followup(result: ToolResult) -> tuple[str, dict] | None:
    """Determine whether a follow-up tool call is needed based on the first tool's result.

    Returns (tool_name, arguments) when a coverage rule says prior approval is
    required and includes a policy document ID. Policy document results and
    terminal coverage rules do not require any follow-up.
    """
    if isinstance(result, CoverageRuleResult) and result.approval_required and result.policy_document_id:
        return "fetch_policy_document", {"document_id": result.policy_document_id}
    return None


def validate_final_summary(raw_summary: dict) -> PriorAuthSummary | StructuredError:
    """Validate the LLM's final summary against the required schema.

    Returns a PriorAuthSummary if valid, or a StructuredError if fields are
    missing or invalid. This guards the auth_summaries contract from missing
    fields such as ``approval_required`` and ``required_documents``.
    """
    if not isinstance(raw_summary, dict):
        return _schema_error(
            "SCHEMA_INVALID_TYPE",
            "Final summary must be a JSON object.",
            {"expected": "object", "actual": type(raw_summary).__name__},
        )

    missing = _missing_fields(raw_summary, PRIOR_AUTH_SUMMARY_REQUIRED_FIELDS)
    if missing:
        return _schema_error(
            "SCHEMA_MISSING_FIELD",
            f"Final summary is missing required field(s): {', '.join(sorted(missing))}",
            {"missing_fields": sorted(missing), "raw_summary": raw_summary},
        )

    err = _validate_bool(raw_summary.get("approval_required"), "approval_required", "final_summary")
    if err:
        return err

    err = _validate_string(raw_summary.get("reason"), "reason", "final_summary")
    if err:
        return err

    err = _validate_string_list(raw_summary.get("required_documents"), "required_documents", "final_summary")
    if err:
        return err

    return PriorAuthSummary(
        approval_required=raw_summary["approval_required"],
        reason=raw_summary["reason"],
        required_documents=list(raw_summary["required_documents"]),
    )


def _error_response(request_id: str, trace: list[dict], error: StructuredError) -> AgentResponse:
    return AgentResponse(summary=None, error=error, trace=trace, request_id=request_id)


def _max_steps_error(trace: list[dict], next_step: str) -> StructuredError:
    return StructuredError(
        code="MAX_STEPS_EXCEEDED",
        message=f"Agent stopped before {next_step}; maximum step count of {config.max_steps} would be exceeded.",
        details={"max_steps": config.max_steps, "trace_length": len(trace), "next_step": next_step},
    )


def _serialize_result(result: ToolResult) -> dict:
    return asdict(result)


def _build_summary_messages(
    member_id: str,
    cpt_code: str,
    query: str,
    parsed_results: list[ToolResult],
) -> list[dict]:
    """Build the prompt for a JSON-only final summary call."""
    payload = {
        "member_id": member_id,
        "cpt_code": cpt_code,
        "query": query,
        "tool_results": [_serialize_result(result) for result in parsed_results],
        "schema": {
            "approval_required": "boolean",
            "reason": "string",
            "required_documents": "array[string]",
        },
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Using the validated tool results below, return only a JSON object matching the schema.\n"
                + json.dumps(payload, sort_keys=True)
            ),
        },
    ]


def run_prior_auth_agent(member_id: str, cpt_code: str, query: str = "") -> AgentResponse:
    """Execute the prior-auth agent workflow with bounded steps.

    1. Call fetch_coverage_rule for the member/procedure
    2. Parse and validate the result
    3. If follow-up is needed, call fetch_policy_document
    4. Call the LLM to synthesize the final summary
    5. Validate and return the response

    The workflow never appends more than ``config.max_steps`` trace entries. Any
    malformed tool output, tool exception, malformed LLM JSON, invalid final
    schema, or step-limit violation is returned as a StructuredError.
    """
    request_id = str(uuid.uuid4())
    trace: list[dict] = []
    parsed_results: list[ToolResult] = []

    def can_take_step(next_step: str) -> StructuredError | None:
        if len(trace) >= config.max_steps:
            return _max_steps_error(trace, next_step)
        return None

    def run_tool_step(tool_name: str, arguments: dict) -> ToolResult | StructuredError:
        step_error = can_take_step(tool_name)
        if step_error:
            return step_error

        trace_entry = {
            "step": len(trace) + 1,
            "tool": tool_name,
            "arguments": arguments,
            "status": "started",
        }
        trace.append(trace_entry)

        try:
            raw_result = tools.execute_tool(tool_name, arguments)
        except Exception as exc:  # pragma: no cover - defensive; tools are deterministic in tests.
            error = StructuredError(
                code="TOOL_EXECUTION_ERROR",
                message=f"Tool {tool_name} raised an exception instead of returning structured data.",
                details={"tool": tool_name, "exception_type": type(exc).__name__, "exception": str(exc)},
            )
            trace_entry.update({"status": "error", "error": asdict(error)})
            return error

        trace_entry["raw_result"] = raw_result
        parsed = parse_tool_result(tool_name, raw_result)
        if isinstance(parsed, StructuredError):
            trace_entry.update({"status": "error", "error": asdict(parsed)})
            return parsed

        trace_entry.update({"status": "ok", "result_type": type(parsed).__name__})
        return parsed

    first_result = run_tool_step(
        "fetch_coverage_rule",
        {"member_id": member_id, "cpt_code": cpt_code},
    )
    if isinstance(first_result, StructuredError):
        return _error_response(request_id, trace, first_result)
    parsed_results.append(first_result)

    # Bounded deterministic follow-up loop. The current schemas only require one
    # follow-up hop, but keeping this as a loop plus a max-step guard prevents
    # accidental infinite iteration if future follow-up rules are added.
    current_result: ToolResult = first_result
    while True:
        followup = should_call_followup(current_result)
        if followup is None:
            break

        followup_tool, followup_args = followup
        followup_result = run_tool_step(followup_tool, followup_args)
        if isinstance(followup_result, StructuredError):
            return _error_response(request_id, trace, followup_result)

        parsed_results.append(followup_result)
        current_result = followup_result

    step_error = can_take_step("llm_summary")
    if step_error:
        return _error_response(request_id, trace, step_error)

    llm_trace = {
        "step": len(trace) + 1,
        "tool": "llm_summary",
        "status": "started",
    }
    trace.append(llm_trace)

    messages = _build_summary_messages(member_id, cpt_code, query, parsed_results)
    try:
        llm_text = llm_client.complete(messages, response_format={"type": "json_object"})
    except Exception as exc:  # pragma: no cover - real providers are not called by tests.
        error = StructuredError(
            code="LLM_ERROR",
            message="LLM summary generation failed.",
            details={"exception_type": type(exc).__name__, "exception": str(exc)},
        )
        llm_trace.update({"status": "error", "error": asdict(error)})
        return _error_response(request_id, trace, error)

    llm_trace["raw_result"] = llm_text
    try:
        raw_summary = json.loads(llm_text)
    except (TypeError, json.JSONDecodeError) as exc:
        error = StructuredError(
            code="LLM_INVALID_JSON",
            message="LLM returned text that could not be parsed as JSON.",
            details={"exception_type": type(exc).__name__, "exception": str(exc), "raw_result": llm_text},
        )
        llm_trace.update({"status": "error", "error": asdict(error)})
        return _error_response(request_id, trace, error)

    summary = validate_final_summary(raw_summary)
    if isinstance(summary, StructuredError):
        llm_trace.update({"status": "error", "error": asdict(summary)})
        return _error_response(request_id, trace, summary)

    llm_trace.update({"status": "ok", "validated_summary": asdict(summary)})
    return AgentResponse(summary=summary, error=None, trace=trace, request_id=request_id)
