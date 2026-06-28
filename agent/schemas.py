"""Typed contracts and validation helpers for tool arguments, results, and final responses."""

from dataclasses import dataclass, field


@dataclass
class CoverageRuleResult:
    member_id: str
    cpt_code: str
    plan_name: str
    approval_required: bool
    reason: str
    policy_document_id: str | None
    required_documents_hint: list[str]
    copay: float | None
    deductible_remaining: float | None


@dataclass
class PolicyDocumentResult:
    document_id: str
    title: str
    summary: str
    required_documents: list[str]
    turnaround_days: int
    submission_method: str


@dataclass
class PriorAuthSummary:
    approval_required: bool
    reason: str
    required_documents: list[str]


@dataclass
class StructuredError:
    code: str
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class AgentResponse:
    summary: PriorAuthSummary | None
    error: StructuredError | None
    trace: list[dict]
    request_id: str


ToolResult = CoverageRuleResult | PolicyDocumentResult


COVERAGE_RULE_REQUIRED_FIELDS = {
    "member_id", "cpt_code", "plan_name", "approval_required", "reason"
}

POLICY_DOCUMENT_REQUIRED_FIELDS = {
    "document_id", "title", "summary", "required_documents"
}

PRIOR_AUTH_SUMMARY_REQUIRED_FIELDS = {
    "approval_required", "reason", "required_documents"
}
