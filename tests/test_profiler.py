from mlverdict.core.config import VerdictConfig
from mlverdict.data.dna import build_dna
from mlverdict.data.profiler import profile_dataset


def test_profile_detects_structure(messy_frame):
    profile = profile_dataset(messy_frame, "churn")
    assert profile.n_rows == len(messy_frame)
    assert "x1" in profile.numeric_columns
    assert "color" in profile.categorical_columns
    assert "const_col" in profile.constant_columns
    assert "row_id" in profile.identifier_columns
    assert profile.n_duplicate_rows >= 2
    assert profile.target.n_unique == 2


def test_dna_is_machine_readable(messy_frame):
    profile = profile_dataset(messy_frame, "churn", VerdictConfig())
    dna = build_dna(profile, group_col=None)
    assert dna.target_name == "churn"
    assert dna.has_duplicates
    assert dna.has_potential_ids
    assert "const_col" in dna.constant_columns
    assert dna.schema_version == "1.0"
