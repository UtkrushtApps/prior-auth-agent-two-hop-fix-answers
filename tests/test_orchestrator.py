"""Candidate-facing tests for the prior-auth agent orchestrator."""

import json
import pytest

from agent import tools
from agent.config import config
from agent.schemas import (
    AgentResponse,
    CoverageRuleResult,
    PolicyDocumentResult,
    PriorAuthSummary,
    StructuredError,
)
from agent.orchestrator import (
    parse_tool_result,
    should_call_followup,
    validate_final_summary,
    run_prior_auth_agent,
)


class TestParseToolResult:
    """Tests for parsing and validating tool outputs."""

    def test_terminal_coverage_rule_parsed(self):
        """Single-hop terminal case: approval not required, no follow-up needed."""
        raw = tools.fetch_coverage_rule("M001", "99213")
        result = parse_tool_result("fetch_coverage_rule", raw)
        assert isinstance(result, CoverageRuleResult)
        assert result.approval_required is False
        assert result.policy_document_id is None

    def test_two_hop_coverage_rule_parsed(self):
        """Two-hop case: approval required, policy document ID present."""
        raw = tools.fetch_coverage_rule("M002", "72148")
        result = parse_tool_result("fetch_coverage_rule", raw)
        assert isinstance(result, CoverageRuleResult)
        assert result.approval_required is True
        assert result.policy_document_id == "DOC-2024-007"

    def test_policy_document_parsed(self):
        """Policy document result is parsed correctly."""
        raw = tools.fetch_policy_document("DOC-2024-007")
        result = parse_tool_result("fetch_policy_document", raw)
        assert isinstance(result, PolicyDocumentResult)
        assert "referral_letter" in result.required_documents

    def test_malformed_coverage_rule_returns_error(self):
        """Malformed tool output (missing required field) yields a StructuredError."""
        raw = tools.fetch_coverage_rule("M004", "80053")
        result = parse_tool_result("fetch_coverage_rule", raw)
        assert isinstance(result, StructuredError)
        assert "SCHEMA" in result.code or "MISSING" in result.code

    def test_not_found_returns_error(self):
        """Tool NOT_FOUND responses become structured errors."""
        raw = tools.fetch_coverage_rule("X999", "00000")
        result = parse_tool_result("fetch_coverage_rule", raw)
        assert isinstance(result, StructuredError)


class TestShouldCallFollowup:
    """Tests for follow-up decision logic."""

    def test_no_followup_when_approval_not_required(self):
        """Terminal result: no second tool call needed."""
        raw = tools.fetch_coverage_rule("M001", "99213")
        result = parse_tool_result("fetch_coverage_rule", raw)
        followup = should_call_followup(result)
        assert followup is None

    def test_followup_when_approval_required(self):
        """Two-hop result: follow-up tool call needed."""
        raw = tools.fetch_coverage_rule("M002", "72148")
        result = parse_tool_result("fetch_coverage_rule", raw)
        followup = should_call_followup(result)
        assert followup is not None
        tool_name, args = followup
        assert tool_name == "fetch_policy_document"
        assert "document_id" in args
        assert args["document_id"] == "DOC-2024-007"

    def test_no_followup_for_policy_document_result(self):
        """Policy document results are terminal — no further follow-up."""
        raw = tools.fetch_policy_document("DOC-2024-007")
        result = parse_tool_result("fetch_policy_document", raw)
        followup = should_call_followup(result)
        assert followup is None


class TestValidateFinalSummary:
    """Tests for LLM output validation."""

    def test_valid_summary_accepted(self):
        raw = {
            "approval_required": True,
            "reason": "MRI requires prior authorization under Silver HMO plan.",
            "required_documents": ["referral_letter", "clinical_notes"]
        }
        result = validate_final_summary(raw)
        assert isinstance(result, PriorAuthSummary)
        assert result.approval_required is True
        assert len(result.required_documents) == 2

    def test_missing_approval_required_rejected(self):
        raw = {
            "reason": "Some reason",
            "required_documents": []
        }
        result = validate_final_summary(raw)
        assert isinstance(result, StructuredError)
        assert "SCHEMA" in result.code or "INVALID" in result.code

    def test_missing_required_documents_rejected(self):
        raw = {
            "approval_required": False,
            "reason": "No auth needed"
        }
        result = validate_final_summary(raw)
        assert isinstance(result, StructuredError)

    def test_wrong_type_rejected(self):
        raw = {
            "approval_required": "yes",
            "reason": "Some reason",
            "required_documents": []
        }
        result = validate_final_summary(raw)
        assert isinstance(result, StructuredError)


class TestRunAgent:
    """Integration tests for the full agent loop (LLM monkeypatched)."""

    def test_single_hop_terminal(self, monkeypatch):
        """Agent does not call fetch_policy_document when approval not required."""
        monkeypatch.setattr(
            "agent.llm_client.complete",
            lambda messages, response_format=None: json.dumps({
                "approval_required": False,
                "reason": "Routine office visit does not require prior authorization.",
                "required_documents": []
            })
        )
        response = run_prior_auth_agent("M001", "99213")
        assert isinstance(response, AgentResponse)
        assert response.error is None
        assert response.summary is not None
        assert response.summary.approval_required is False
        tool_names = [t["tool"] for t in response.trace]
        assert "fetch_policy_document" not in tool_names

    def test_two_hop_with_policy_document(self, monkeypatch):
        """Agent calls fetch_policy_document when approval is required."""
        monkeypatch.setattr(
            "agent.llm_client.complete",
            lambda messages, response_format=None: json.dumps({
                "approval_required": True,
                "reason": "MRI requires prior authorization. Submit referral letter and clinical notes.",
                "required_documents": ["referral_letter", "clinical_notes"]
            })
        )
        response = run_prior_auth_agent("M002", "72148")
        assert isinstance(response, AgentResponse)
        assert response.error is None
        assert response.summary.approval_required is True
        tool_names = [t["tool"] for t in response.trace]
        assert "fetch_policy_document" in tool_names

    def test_malformed_tool_output_returns_error(self):
        """Malformed coverage rule data produces a structured error, not a crash."""
        response = run_prior_auth_agent("M004", "80053")
        assert isinstance(response, AgentResponse)
        assert response.error is not None
        assert response.summary is None

    def test_max_steps_not_exceeded(self, monkeypatch):
        """Agent does not exceed the configured max steps."""
        monkeypatch.setattr(
            "agent.llm_client.complete",
            lambda messages, response_format=None: json.dumps({
                "approval_required": True,
                "reason": "Test reason",
                "required_documents": []
            })
        )
        response = run_prior_auth_agent("M002", "72148")
        assert isinstance(response, AgentResponse)
        assert len(response.trace) <= config.max_steps
