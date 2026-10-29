import json

import numpy as np
import pytest
import torch

from canonical_evaluation import EvaluationConfig, evaluate_policy, write_results
from game import Game2048
from legacy_agent import (
    EXPECTED_KEYS,
    EXPECTED_SHAPES,
    LEGACY_ACTION_MAPPING,
    LegacyDQNAgent,
    LegacyQNetwork,
    action_mapping_metadata,
)


CHECKPOINT = "backups/strategy_dqn_v1.pt"


def test_actual_strategy_v1_checkpoint_loads():
    agent = LegacyDQNAgent.load(CHECKPOINT, device=torch.device("cpu"))

    assert isinstance(agent.network, LegacyQNetwork)
    assert agent.network.net[0].in_features == 16
    assert agent.network.net[-1].out_features == 4


def test_malformed_checkpoint_keys_are_rejected(tmp_path):
    path = tmp_path / "bad_keys.pt"
    torch.save({"online": {"wrong": torch.zeros(1)}}, path)

    with pytest.raises(ValueError, match="incompatible legacy online keys"):
        LegacyDQNAgent.load(str(path), device=torch.device("cpu"))


def test_malformed_checkpoint_shapes_are_rejected(tmp_path):
    online = {
        key: torch.zeros(shape)
        for key, shape in EXPECTED_SHAPES.items()
    }
    online["net.0.weight"] = torch.zeros(128, 16)
    path = tmp_path / "bad_shape.pt"
    torch.save({"online": online}, path)

    with pytest.raises(ValueError, match="net.0.weight.*shape"):
        LegacyDQNAgent.load(str(path), device=torch.device("cpu"))


def test_dueling_architecture_is_rejected(tmp_path):
    path = tmp_path / "wrong_architecture.pt"
    torch.save(
        {
            "architecture": "dueling_double_dqn",
            "online": {
                key: torch.zeros(shape)
                for key, shape in EXPECTED_SHAPES.items()
            },
        },
        path,
    )

    with pytest.raises(ValueError, match="architecture"):
        LegacyDQNAgent.load(str(path), device=torch.device("cpu"))


def test_state_dimensions_and_action_mapping():
    agent = LegacyDQNAgent.load(CHECKPOINT, device=torch.device("cpu"))
    env = Game2048(seed=51)
    state = env.state_vector()

    assert state.shape == (16,)
    assert LEGACY_ACTION_MAPPING == {
        0: "up", 1: "right", 2: "down", 3: "left"
    }
    assert action_mapping_metadata()["action_mapping_status"] == (
        "assumed_unverified"
    )
    assert agent.act(state, valid_actions=env.valid_actions()) in env.valid_actions()


def test_legal_action_masking_and_determinism():
    network = LegacyQNetwork()
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.net[-1].bias.copy_(torch.tensor([1.0, 4.0, 3.0, 2.0]))
    agent = LegacyDQNAgent(network, device=torch.device("cpu"))
    state = np.zeros(16, dtype=np.float32)

    actions = [
        agent.act(state, valid_actions=[0, 2])
        for _ in range(5)
    ]

    assert actions == [2] * 5
    assert agent.act(state, valid_actions=[]) is None


def test_canonical_evaluation_with_legacy_agent(tmp_path):
    agent = LegacyDQNAgent.load(CHECKPOINT, device=torch.device("cpu"))
    rows, metadata = evaluate_policy(
        agent,
        EvaluationConfig(seeds=(61, 62), max_moves=10),
        agent_id="legacy_strategy_v1",
        checkpoint_path=CHECKPOINT,
        architecture="legacy_feedforward_dqn",
    )
    metadata.update(action_mapping_metadata())
    csv_path, json_path = write_results(
        str(tmp_path), rows, metadata
    )

    with open(json_path) as handle:
        document = json.load(handle)

    assert len(rows) == 2
    assert all(row["agent_id"] == "legacy_strategy_v1" for row in rows)
    assert document["metadata"]["checkpoint_path"] == CHECKPOINT
    assert document["metadata"]["action_mapping_status"] == (
        "assumed_unverified"
    )
    assert csv_path.endswith("games.csv")
