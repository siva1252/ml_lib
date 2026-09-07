"""MLVerdict — Evidence-Based Machine Learning Decision Engine."""

from mlverdict.api.automl import AutoML
from mlverdict.api.run import Run
from mlverdict.api.verdict import Verdict
from mlverdict.core.enums import DecisionStatus, ProblemType, Severity, UnsupervisedTask
from mlverdict.production.artifact import ModelArtifact

__version__ = "0.2.0"
__all__ = [
    "Verdict",
    "AutoML",
    "Run",
    "ModelArtifact",
    "DecisionStatus",
    "ProblemType",
    "Severity",
    "UnsupervisedTask",
    "__version__",
]
