from types import SimpleNamespace

import pytest
import torch

from train import (
    checkpoint_metadata,
    default_output_dir,
    resolve_output_dir,
    resolve_resume_paths,
    validate_output_directory,
    validate_resume_checkpoint,
)


def test_reward_mode_output_directories():
    assert default_output_dir("shaped") == "checkpoints_dueling"
    assert default_output_dir("raw_score") == "checkpoints_dueling_raw_score"
    assert resolve_output_dir("raw_score") == "checkpoints_dueling_raw_score"
    assert resolve_output_dir("shaped", "custom-output") == "custom-output"


def test_invalid_training_reward_mode_is_rejected():
    with pytest.raises(ValueError, match="reward_mode"):
        default_output_dir("invalid")


def test_checkpoint_metadata_records_reward_mode():
    args = SimpleNamespace(reward_mode="raw_score", out=None)
    metadata = checkpoint_metadata(args, "raw_score")
    assert metadata["reward_mode"] == "raw_score"
    assert metadata["model"] == "dueling_double_dqn"


def test_resume_requires_matching_explicit_reward_metadata(tmp_path):
    missing = tmp_path / "missing.pt"
    torch.save({"metadata": {}}, missing)
    with pytest.raises(ValueError, match="missing explicit reward_mode"):
        validate_resume_checkpoint(str(missing), "raw_score")

    mismatch = tmp_path / "mismatch.pt"
    torch.save({"metadata": {"reward_mode": "shaped"}}, mismatch)
    with pytest.raises(ValueError, match="does not match"):
        validate_resume_checkpoint(str(mismatch), "raw_score")

    matching = tmp_path / "matching.pt"
    torch.save({"metadata": {"reward_mode": "raw_score"}}, matching)
    assert validate_resume_checkpoint(str(matching), "raw_score")["metadata"][
        "reward_mode"
    ] == "raw_score"


def test_resume_inside_selected_output_directory_is_valid(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    checkpoint = output_dir / "dqn_2048.pt"
    torch.save({"metadata": {"reward_mode": "raw_score"}}, checkpoint)

    selected, resolved_checkpoint = resolve_resume_paths(
        "raw_score", str(output_dir), str(checkpoint)
    )

    assert selected == str(output_dir.resolve())
    assert resolved_checkpoint == str(checkpoint.resolve())
    validate_output_directory(selected, resolved_checkpoint)
    validate_resume_checkpoint(resolved_checkpoint, "raw_score")


def test_resume_outside_selected_output_directory_is_rejected(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint-run"
    selected_dir = tmp_path / "other-run"
    checkpoint_dir.mkdir()
    selected_dir.mkdir()
    checkpoint = checkpoint_dir / "dqn_2048.pt"
    torch.save({"metadata": {"reward_mode": "raw_score"}}, checkpoint)

    with pytest.raises(ValueError, match="inside the selected output"):
        resolve_resume_paths(
            "raw_score", str(selected_dir), str(checkpoint)
        )


def test_resume_targeting_unrelated_existing_directory_is_rejected(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint-run"
    unrelated_dir = tmp_path / "unrelated-run"
    checkpoint_dir.mkdir()
    unrelated_dir.mkdir()
    checkpoint = checkpoint_dir / "dqn_2048.pt"
    torch.save({"metadata": {"reward_mode": "shaped"}}, checkpoint)

    with pytest.raises(ValueError, match="inside the selected output"):
        resolve_resume_paths(
            "shaped", str(unrelated_dir), str(checkpoint)
        )


def test_fresh_run_existing_output_directory_is_rejected(tmp_path):
    output_dir = tmp_path / "existing-run"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        validate_output_directory(str(output_dir))