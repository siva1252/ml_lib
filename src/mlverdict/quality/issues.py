"""Quality-issue constructors."""

from __future__ import annotations

from mlverdict.core.enums import Severity
from mlverdict.core.types import QualityIssue


def issue(
    issue_type: str,
    severity: Severity,
    description: str,
    evidence: str,
    recommendation: str,
    column: str | None = None,
) -> QualityIssue:
    return QualityIssue(
        issue_type=issue_type,
        severity=severity,
        column=column,
        evidence=evidence,
        description=description,
        recommendation=recommendation,
    )
