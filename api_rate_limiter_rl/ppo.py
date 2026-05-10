from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
except Exception as exc:  # pragma: no cover - import guard for non-torch envs
    torch = None
    nn = None
    F = None
    Categorical = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


@dataclass
class RolloutBatch:
    states: np.ndarray
    actions: np.ndarray
    old_logp: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    values: np.ndarray


if torch is not None:

    def _resolve_torch_device(device: str | None):
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    class ActorCriticNet(nn.Module):
        def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            self.actor = nn.Linear(hidden_dim, action_dim)
            self.critic = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            h = self.shared(x)
            return self.actor(h), self.critic(h).squeeze(-1)


    class PPOAgent:
        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int = 48,
            actor_lr: float = 0.005,
            critic_lr: float = 0.008,
            clip_eps: float = 0.2,
            gamma: float = 0.97,
            lam: float = 0.93,
            update_epochs: int = 8,
            minibatch_size: int = 128,
            seed: int = 7,
            device: str | None = None,
        ):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.clip_eps = clip_eps
            self.gamma = gamma
            self.lam = lam
            self.update_epochs = update_epochs
            self.minibatch_size = minibatch_size
            self.device = _resolve_torch_device(device)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            self.model = ActorCriticNet(state_dim, hidden_dim, action_dim).to(self.device)
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.model.shared.parameters(), "lr": critic_lr},
                    {"params": self.model.actor.parameters(), "lr": actor_lr},
                    {"params": self.model.critic.parameters(), "lr": critic_lr},
                ],
            )

        def act(self, state: Sequence[float], deterministic: bool = False):
            x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                logits, value = self.model(x)
                dist = Categorical(logits=logits)
                if deterministic:
                    action = int(torch.argmax(dist.probs, dim=-1).item())
                else:
                    action = int(dist.sample().item())
                logp = float(dist.log_prob(torch.tensor(action, device=self.device)).item())
            return action, logp, float(value.item()), dist.probs.squeeze(0).detach().cpu().tolist()

        def compute_gae(self, rewards, values, dones, last_value):
            advantages = np.zeros_like(rewards, dtype=float)
            gae = 0.0
            for t in reversed(range(len(rewards))):
                next_value = last_value if t == len(rewards) - 1 else values[t + 1]
                next_non_terminal = 1.0 - float(dones[t])
                delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
                gae = delta + self.gamma * self.lam * next_non_terminal * gae
                advantages[t] = gae
            returns = advantages + values
            return advantages, returns

        def _iterate_minibatches(self, size: int):
            indices = np.random.permutation(size)
            for start in range(0, size, self.minibatch_size):
                yield indices[start : start + self.minibatch_size]

        def update(self, batch: RolloutBatch, behavior_batch=None, behavior_weight: float = 0.0):
            states = torch.tensor(batch.states, dtype=torch.float32, device=self.device)
            actions = torch.tensor(batch.actions, dtype=torch.int64, device=self.device)
            old_logp = torch.tensor(batch.old_logp, dtype=torch.float32, device=self.device)
            returns = torch.tensor(batch.returns, dtype=torch.float32, device=self.device)
            advantages = torch.tensor(batch.advantages, dtype=torch.float32, device=self.device)
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            bc_states = None
            bc_actions = None
            if behavior_batch is not None and behavior_weight > 0.0:
                bc_states = torch.tensor(behavior_batch["states"], dtype=torch.float32, device=self.device)
                bc_actions = torch.tensor(behavior_batch["actions"], dtype=torch.int64, device=self.device)

            for _ in range(self.update_epochs):
                for mb_idx in self._iterate_minibatches(len(states)):
                    mb_idx = torch.tensor(mb_idx, dtype=torch.int64, device=self.device)
                    mb_states = states[mb_idx]
                    mb_actions = actions[mb_idx]
                    mb_old_logp = old_logp[mb_idx]
                    mb_adv = advantages[mb_idx]
                    mb_returns = returns[mb_idx]

                    logits, values = self.model(mb_states)
                    dist = Categorical(logits=logits)
                    new_logp = dist.log_prob(mb_actions)
                    ratio = torch.exp(new_logp - mb_old_logp)
                    unclipped = ratio * mb_adv
                    clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                    actor_loss = -torch.mean(torch.min(unclipped, clipped))
                    critic_loss = F.mse_loss(values, mb_returns)
                    entropy_bonus = dist.entropy().mean()
                    behavior_bonus = torch.tensor(0.0, device=self.device)
                    if bc_states is not None and bc_actions is not None:
                        sample_size = min(len(mb_idx), len(bc_states))
                        bc_idx = torch.randint(0, len(bc_states), (sample_size,), device=self.device)
                        bc_logits, _ = self.model(bc_states[bc_idx])
                        bc_dist = Categorical(logits=bc_logits)
                        behavior_bonus = bc_dist.log_prob(bc_actions[bc_idx]).mean()
                    loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus - behavior_weight * behavior_bonus

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()

        def batch_from_rollout(self, states, actions, logps, rewards, values, dones, last_value):
            advantages, returns = self.compute_gae(np.asarray(rewards), np.asarray(values), np.asarray(dones), last_value)
            return RolloutBatch(
                states=np.asarray(states, dtype=float),
                actions=np.asarray(actions, dtype=int),
                old_logp=np.asarray(logps, dtype=float),
                returns=np.asarray(returns, dtype=float),
                advantages=np.asarray(advantages, dtype=float),
                values=np.asarray(values, dtype=float),
            )

        def checkpoint(self):
            return {
                "model": copy.deepcopy(self.model.state_dict()),
                "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            }

        def restore(self, checkpoint):
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        def set_learning_rates(self, actor_lr: float, critic_lr: float):
            self.optimizer.param_groups[0]["lr"] = critic_lr
            self.optimizer.param_groups[1]["lr"] = actor_lr
            self.optimizer.param_groups[2]["lr"] = critic_lr

        def save(self, path: str | Path):
            torch.save(
                {
                    "state_dim": self.state_dim,
                    "action_dim": self.action_dim,
                    "checkpoint": self.checkpoint(),
                },
                path,
            )

        def load(self, path: str | Path):
            payload = torch.load(path, map_location=self.device)
            self.restore(payload["checkpoint"])

else:

    class PPOAgent:  # pragma: no cover - import-time fallback only
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for PPOAgent. Install torch and rerun."
            ) from _TORCH_IMPORT_ERROR
