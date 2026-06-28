"""Deterministic local tool implementations for prior-auth lookup."""

import json
from agent.config import FIXTURES_DIR

_fixture_cache: dict | None = None


def _load_fixture() -> dict:
    global _fixture_cache
    if _fixture_cache is None:
        with open(FIXTURES_DIR / "tool_data.json") as f:
            _fixture_cache = json.load(f)
    return _fixture_cache


def fetch_coverage_rule(member_id: str, cpt_code: str) -> dict:
    """Look up coverage rule for a member and procedure code.

    Args:
        member_id: The member identifier (e.g. "M001").
        cpt_code: The CPT procedure code (e.g. "99213").

    Returns:
        Coverage rule dict or an error dict with error_code.
    """
    key = f"{member_id}-{cpt_code}"
    data = _load_fixture()
    rule = data.get("coverage_rules", {}).get(key)
    if rule is None:
        return {"error_code": "NOT_FOUND", "message": f"No coverage rule found for member {member_id}, CPT {cpt_code}"}
    return rule


def fetch_policy_document(document_id: str) -> dict:
    """Fetch a policy document by its ID.

    Args:
        document_id: The policy document identifier (e.g. "DOC-2024-007").

    Returns:
        Policy document dict or an error dict with error_code.
    """
    data = _load_fixture()
    doc = data.get("policy_documents", {}).get(document_id)
    if doc is None:
        return {"error_code": "NOT_FOUND", "message": f"No policy document found for {document_id}"}
    return doc


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_coverage_rule",
            "description": "Look up coverage rule for a member and CPT procedure code. Returns approval requirements and policy references.",
            "parameters": {
                "type": "object",
                "properties": {
                    "member_id": {"type": "string", "description": "Member identifier, e.g. 'M001'"},
                    "cpt_code": {"type": "string", "description": "CPT procedure code, e.g. '99213'"}
                },
                "required": ["member_id", "cpt_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_policy_document",
            "description": "Fetch detailed policy document by document ID. Returns required documents, turnaround time, and submission method.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "Policy document identifier, e.g. 'DOC-2024-007'"}
                },
                "required": ["document_id"]
            }
        }
    }
]


def execute_tool(tool_name: str, arguments: dict) -> dict:
    """Execute a tool by name with the given arguments."""
    if tool_name == "fetch_coverage_rule":
        return fetch_coverage_rule(**arguments)
    elif tool_name == "fetch_policy_document":
        return fetch_policy_document(**arguments)
    else:
        return {"error_code": "UNKNOWN_TOOL", "message": f"Unknown tool: {tool_name}"}
