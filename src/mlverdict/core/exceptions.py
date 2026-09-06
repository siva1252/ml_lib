"""Library exceptions. Prefer returning UNDECIDED/BLOCKED on a Run when possible."""

from __future__ import annotations


class VerdictError(Exception):
    """Base error for MLVerdict."""


class TargetNotFoundError(VerdictError):
    """Target column is missing from the dataset."""


class DataLoadError(VerdictError):
    """Dataset could not be loaded."""


class ArtifactError(VerdictError):
    """Save, load, or prediction against an artifact failed."""


class ConfigurationError(VerdictError):
    """User configuration is invalid."""
