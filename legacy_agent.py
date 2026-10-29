"""Inference adapter for the legacy Strategy V1 DQN checkpoint."""

from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn


LEGACY_ACTIONS = ("up", "right", "down", "left")
LEGACY_ACTION_MAPPING = {
    index: action for index, action in enumerate(LEGACY_ACTIONS)
}
LEGACY_ARCHITECTURE = "legacy_feedforward_dqn"
EXPECTED_KEYS = (
    "net.0.weight",
    "net.0.bias",
    "net.2.weight",
    "net.2.bias",
    "net.4.weight",
    "net.4.bias",
)
EXPECTED_SHAPES = {
    "net.0.weight": (256, 16),
    "net.0.bias": (256,),
    "net.2.weight": (256, 256),
    "net.2.bias": (256,),
    "net.4.weight": (4, 256),
    "net.4.bias": (4,),
}


class LegacyQNetwork(nn.Module):
    """The non-dueling feedforward network used by Strategy V1."""

    def __init__(self, state_size=16, action_size=4):
        super().__init__()
        if state_size != 16 or action_size != 4:
            raise ValueError(
                "legacy network requires state_size=16 and action_size=4"
            )
        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

    def forward(self, state):
        return self.net(state)


def _validate_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("legacy checkpoint must contain a dictionary")
    if "online" not in checkpoint:
        raise ValueError("legacy checkpoint is missing the 'online' weights")

    online = checkpoint["online"]
    if not isinstance(online, dict):
        raise ValueError("legacy checkpoint 'online' value must be a state dict")

    actual_keys = tuple(online.keys())
    if set(actual_keys) != set(EXPECTED_KEYS):
        missing = sorted(set(EXPECTED_KEYS) - set(actual_keys))
        unexpected = sorted(set(actual_keys) - set(EXPECTED_KEYS))
        raise ValueError(
            "incompatible legacy online keys; "
            f"missing={missing}, unexpected={unexpected}"
        )

    for key, expected_shape in EXPECTED_SHAPES.items():
        value = online[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"legacy weight {key} is not a tensor")
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"legacy weight {key} has shape {tuple(value.shape)}; "
                f"expected {expected_shape}"
            )

    architecture = checkpoint.get("architecture")
    if architecture not in (None, LEGACY_ARCHITECTURE):
        raise ValueError(
            f"checkpoint architecture {architecture!r} is not "
            f"{LEGACY_ARCHITECTURE!r}"
        )

    return online


class LegacyDQNAgent:
    """Deterministic, legally masked inference for a legacy checkpoint."""

    def __init__(self, network, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.network = network.to(self.device)
        self.network.eval()

    @classmethod
    def load(cls, path, device=None):
        selected_device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        checkpoint = torch.load(
            path,
            map_location=selected_device,
            weights_only=False,
        )
        online = _validate_checkpoint(checkpoint)
        network = LegacyQNetwork()
        network.load_state_dict(online, strict=True)
        return cls(network, selected_device)

    def act(self, state, explore=False, valid_actions=None):
        """Return the highest-Q legal action, or None if none are legal."""
        del explore
        actions = list(range(4)) if valid_actions is None else list(valid_actions)
        if not actions:
            return None
        if any(action not in range(4) for action in actions):
            raise ValueError(f"invalid action list: {actions}")

        values = np.asarray(state, dtype=np.float32)
        if values.shape != (16,):
            raise ValueError(
                f"legacy state must have shape (16,), got {values.shape}"
            )

        with torch.no_grad():
            tensor = torch.as_tensor(
                values, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values = self.network(tensor).squeeze(0)
            invalid = torch.ones(4, dtype=torch.bool, device=self.device)
            invalid[actions] = False
            q_values = q_values.masked_fill(invalid, float("-inf"))
            return int(q_values.argmax().item())

    def __call__(self, state, valid_actions):
        return self.act(state, explore=False, valid_actions=valid_actions)


def action_mapping_metadata():
    """Describe the assumed, historically unverified output mapping."""
    return {
        "action_mapping": {
            str(index): action
            for index, action in LEGACY_ACTION_MAPPING.items()
        },
        "action_mapping_status": "assumed_unverified",
        "action_mapping_basis": (
            "Uses the current Game2048 ordering; the legacy checkpoint does "
            "not serialize action labels."
        ),
    }
