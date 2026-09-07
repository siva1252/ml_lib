"""MLVerdict — Evidence-Based Machine Learning Decision Engine."""

from mlverdict.api.automl import AutoML
from mlverdict.api.run import Run
from mlverdict.api.verdict import Verdict
from mlverdict.core.enums import DecisionStatus, ProblemType, Severity
from mlverdict.production.artifact import ModelArtifact

__version__ = "0.1.3"
__all__ = [
    "Verdict",
    "AutoML",
    "Run",
    "ModelArtifact",
    "DecisionStatus",
    "ProblemType",
    "Severity",
    "__version__",
]
