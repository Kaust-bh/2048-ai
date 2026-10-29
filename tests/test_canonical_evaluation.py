import json

import pytest

from canonical_evaluation import (
    EvaluationConfig,
    evaluate_policy,
    read_results,
    summary_from_csv,
    write_results,
)


class FirstValidPolicy:
    def __call__(self, state, valid_actions):
        assert state.shape == (16,)
        return valid_actions[0]


class InvalidPolicy:
    def __call__(self, state, valid_actions):
        return 99


def test_same_seed_list_is_reproducible():
    config = EvaluationConfig(seeds=(11, 12), max_moves=40)
    first, first_metadata = evaluate_policy(
        FirstValidPolicy(), config, agent_id="fake"
    )
    second, second_metadata = evaluate_policy(
        FirstValidPolicy(), config, agent_id="fake"
    )

    assert first == second
    assert first_metadata["evaluation"] == second_metadata["evaluation"]


def test_invalid_action_is_rejected_before_step():
    config = EvaluationConfig(seeds=(11,), max_moves=2)

    with pytest.raises(ValueError, match="invalid action"):
        evaluate_policy(InvalidPolicy(), config, agent_id="invalid")


def test_csv_round_trip_and_summary_use_saved_rows(tmp_path):
    config = EvaluationConfig(seeds=(21, 22, 23), max_moves=30)
    rows, metadata = evaluate_policy(
        FirstValidPolicy(),
        config,
        agent_id="fake",
        checkpoint_path="example.pt",
        architecture="fake_architecture",
    )
    csv_path, json_path = write_results(
        str(tmp_path), rows, metadata
    )

    loaded_rows = read_results(csv_path)
    loaded_summary = summary_from_csv(csv_path)
    with open(json_path) as handle:
        document = json.load(handle)

    assert loaded_rows == rows
    assert document["summary"] == loaded_summary
    assert "reached_512_pct" in loaded_summary
    assert "reached_1024_pct" in loaded_summary
    assert document["metadata"]["checkpoint_path"] == "example.pt"
    assert document["metadata"]["architecture"] == "fake_architecture"
    assert all(row["agent_id"] == "fake" for row in loaded_rows)
    assert all("seed" in row for row in loaded_rows)
    assert all("reached_2048" in row for row in loaded_rows)


def test_summary_statistics_match_per_game_values(tmp_path):
    config = EvaluationConfig(seeds=(31, 32), max_moves=1)
    rows, metadata = evaluate_policy(
        FirstValidPolicy(), config, agent_id="fake"
    )
    rows[0]["score"] = 10
    rows[0]["max_tile"] = 2048
    rows[0]["moves"] = 1
    rows[0]["reached_2048"] = True
    rows[1]["score"] = 30
    rows[1]["max_tile"] = 4
    rows[1]["moves"] = 0
    rows[1]["reached_2048"] = False
    csv_path, _ = write_results(str(tmp_path), rows, metadata)

    summary = summary_from_csv(csv_path)

    assert summary["mean_score"] == 20.0
    assert summary["median_score"] == 20.0
    assert summary["score_stddev"] == 10.0
    assert summary["mean_moves"] == 0.5
    assert summary["max_tile_distribution"] == {"4": 1, "2048": 1}
    assert summary["reached_2048_pct"] == 50.0


def test_optional_callbacks_report_completion_and_progress():
    completed = []
    progress = []

    evaluate_policy(
        FirstValidPolicy(),
        EvaluationConfig(seeds=(71, 72, 73), max_moves=2),
        agent_id="fake",
        on_game_complete=lambda row: completed.append(row["game"]),
        on_progress=lambda done, total: progress.append((done, total)),
        progress_interval=2,
    )

    assert completed == [1, 2, 3]
    assert progress == [(2, 3), (3, 3)]


def test_callback_exceptions_propagate():
    def fail(_row):
        raise RuntimeError("callback failure")

    with pytest.raises(RuntimeError, match="callback failure"):
        evaluate_policy(
            FirstValidPolicy(),
            EvaluationConfig(seeds=(74,), max_moves=1),
            agent_id="fake",
            on_game_complete=fail,
        )
