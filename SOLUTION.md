# Solution Steps

1. Import the schema dataclasses and required-field constants into agent/orchestrator.py so tool outputs and final summaries can be validated against the same contracts used by the rest of the app.

2. Add small validation helpers for missing fields, primitive type checks, string-list checks, optional numeric fields, and consistently shaped StructuredError objects.

3. Implement parse_tool_result so it accepts only known tools, rejects non-dict outputs, converts tool error_code payloads into StructuredError, checks required fields, validates field types, and safely constructs CoverageRuleResult or PolicyDocumentResult dataclasses.

4. Implement should_call_followup so a CoverageRuleResult triggers fetch_policy_document only when approval_required is true and policy_document_id is present; all terminal coverage rules and policy document results return None.

5. Implement validate_final_summary so the LLM JSON object must include approval_required, reason, and required_documents with the correct boolean/string/list-of-strings types before constructing PriorAuthSummary.

6. Implement run_prior_auth_agent as a bounded deterministic workflow: fetch coverage rule, validate it, optionally fetch and validate the policy document, then call the LLM once to synthesize the summary.

7. Track each tool/LLM action in trace with step number, tool name, status, arguments or raw result where appropriate, and error details on failure.

8. Before every tool or LLM step, check len(trace) against config.max_steps; if the next step would exceed the bound, return an AgentResponse with a MAX_STEPS_EXCEEDED StructuredError without appending more trace entries.

9. Wrap tool execution, LLM execution, JSON parsing, and schema validation in structured error handling so malformed outputs or exceptions return AgentResponse(error=...) instead of crashing.

10. Build the LLM prompt from validated dataclass results only, request JSON output with response_format={"type": "json_object"}, parse the returned JSON, validate it with validate_final_summary, and return AgentResponse(summary=..., error=None, trace=..., request_id=...).

