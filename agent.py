
"""Dueling Double DQN agent for 2048 with valid-action masking."""

from collections import deque
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    """Estimate state value and action advantages separately."""

    def __init__(self, state_size=16, action_size=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x):
        features = self.features(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        return value + advantage - advantage.mean(
            dim=1, keepdim=True
        )


# Compatibility alias.
QNetwork = DuelingQNetwork


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.memory = deque(maxlen=capacity)

    def add(
        self, state, action, reward, next_state,
        done, next_valid_actions
    ):
        mask = np.zeros(4, dtype=bool)

        if next_valid_actions:
            mask[list(next_valid_actions)] = True

        self.memory.append((
            np.asarray(state, dtype=np.float32).copy(),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32).copy(),
            float(done),
            mask,
        ))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones, masks = zip(*batch)

        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            np.asarray(masks, dtype=bool),
        )

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(
        self,
        state_size=16,
        action_size=4,
        learning_rate=3e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9995,
        buffer_size=100_000,
        batch_size=128,
        target_update=1000,
        seed=42,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps = 0

        self.online = DuelingQNetwork(
            state_size, action_size
        ).to(self.device)

        self.target = DuelingQNetwork(
            state_size, action_size
        ).to(self.device)

        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=learning_rate,
        )

        self.memory = ReplayBuffer(buffer_size)

    def act(self, state, explore=True, valid_actions=None):
        if valid_actions is None:
            valid_actions = list(range(self.action_size))
        else:
            valid_actions = list(valid_actions)

        if not valid_actions:
            return 0

        if explore and random.random() < self.epsilon:
            return random.choice(valid_actions)

        with torch.no_grad():
            x = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            q_values = self.online(x).squeeze(0)

            invalid_mask = torch.ones(
                self.action_size,
                dtype=torch.bool,
                device=self.device,
            )
            invalid_mask[valid_actions] = False

            q_values = q_values.masked_fill(
                invalid_mask, float("-inf")
            )

        return int(q_values.argmax().item())

    def remember(
        self, state, action, reward, next_state,
        done, next_valid_actions
    ):
        self.memory.add(
            state, action, reward, next_state,
            done, next_valid_actions
        )

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            next_masks,
        ) = self.memory.sample(self.batch_size)

        states = torch.as_tensor(
            states, dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            actions, dtype=torch.long, device=self.device
        )
        rewards = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            dones, dtype=torch.float32, device=self.device
        )
        next_masks = torch.as_tensor(
            next_masks, dtype=torch.bool, device=self.device
        )

        # Q-value for the action actually taken.
        current_q = self.online(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            # Double DQN: online network chooses the next action.
            online_next_q = self.online(next_states)
            online_next_q = online_next_q.masked_fill(
                ~next_masks, float("-inf")
            )

            no_valid_moves = ~next_masks.any(dim=1)
            next_actions = online_next_q.argmax(dim=1)

            # Target network evaluates that selected action.
            target_next_q = self.target(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)

            target_next_q = target_next_q.masked_fill(
                no_valid_moves, 0.0
            )

            target_q = (
                rewards
                + self.gamma * target_next_q * (1.0 - dones)
            )

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), max_norm=10.0
        )

        self.optimizer.step()
        self.steps += 1

        if self.steps % self.target_update == 0:
            self.target.load_state_dict(
                self.online.state_dict()
            )

        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    def save(self, path, episode=0, metadata=None):
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
            "episode": episode,
            "metadata": metadata or {},
            "architecture": "dueling_double_dqn",
        }, path)

    def load(self, path, load_optimizer=False):
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(
            checkpoint.get("target", checkpoint["online"])
        )

        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(
                checkpoint["optimizer"]
            )

        self.epsilon = float(
            checkpoint.get("epsilon", self.epsilon)
        )
        self.steps = int(checkpoint.get("steps", 0))

        return checkpoint
