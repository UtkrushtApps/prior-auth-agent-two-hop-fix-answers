"""CLI and selfcheck entrypoint."""

import json
import sys

from agent.config import config, FIXTURES_DIR


def selfcheck() -> int:
    """Validate scaffold readiness without calling candidate stubs or requiring API keys."""
    errors = []

    fixture_path = FIXTURES_DIR / "tool_data.json"
    if not fixture_path.exists():
        errors.append(f"Missing fixture file: {fixture_path}")
    else:
        try:
            with open(fixture_path) as f:
                data = json.load(f)
            if "coverage_rules" not in data:
                errors.append("Fixture missing 'coverage_rules' key")
            if "policy_documents" not in data:
                errors.append("Fixture missing 'policy_documents' key")
            if len(data.get("coverage_rules", {})) < 3:
                errors.append("Expected at least 3 coverage rule entries")
            if len(data.get("policy_documents", {})) < 2:
                errors.append("Expected at least 2 policy document entries")
        except json.JSONDecodeError as e:
            errors.append(f"Invalid fixture JSON: {e}")

    if not config.model:
        errors.append("AGENT_MODEL not configured")

    try:
        from agent import tools, schemas, llm_client
    except ImportError as e:
        errors.append(f"Import error: {e}")

    try:
        from agent.tools import TOOL_DEFINITIONS
        if len(TOOL_DEFINITIONS) < 2:
            errors.append("Expected at least 2 tool definitions")
    except Exception as e:
        errors.append(f"Tool definition error: {e}")

    try:
        from agent.schemas import (
            CoverageRuleResult, PolicyDocumentResult,
            PriorAuthSummary, StructuredError, AgentResponse,
            COVERAGE_RULE_REQUIRED_FIELDS,
            POLICY_DOCUMENT_REQUIRED_FIELDS,
            PRIOR_AUTH_SUMMARY_REQUIRED_FIELDS,
        )
    except ImportError as e:
        errors.append(f"Schema import error: {e}")

    if errors:
        for e in errors:
            print(f"  FAIL: {e}", file=sys.stderr)
        return 1

    print("Selfcheck passed: fixtures, config, and imports OK")
    return 0


def main() -> int:
    if "--selfcheck" in sys.argv:
        return selfcheck()

    if not config.has_api_key:
        print("Error: Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env", file=sys.stderr)
        return 1

    from agent.orchestrator import run_prior_auth_agent

    member_id = "M002"
    cpt_code = "72148"
    query = "Does member M002 need prior authorization for CPT 72148?"

    print(f"Running prior-auth agent for member={member_id}, cpt={cpt_code}...")
    response = run_prior_auth_agent(member_id, cpt_code, query)

    print(f"\nRequest ID: {response.request_id}")
    if response.error:
        print(f"Error [{response.error.code}]: {response.error.message}")
    else:
        print(f"Approval Required: {response.summary.approval_required}")
        print(f"Reason: {response.summary.reason}")
        print(f"Required Documents: {', '.join(response.summary.required_documents)}")

    print(f"\nTrace ({len(response.trace)} steps):")
    for entry in response.trace:
        print(f"  Step {entry.get('step')}: {entry.get('tool')} -> {entry.get('status')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
