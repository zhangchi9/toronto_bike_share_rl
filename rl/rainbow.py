"""Round 11: Rainbow-style DQN (PyTorch, no stable-baselines3).

DQN plus four of Rainbow's parts (Hessel et al. 2018):
  - double Q-learning   the online net picks the next action, the target net values it
  - dueling network     Q = V + A - mean(A), shared [256, 256] body
  - prioritized replay  proportional, alpha 0.6, beta 0.4 -> 1 over training
  - n-step returns      3 steps
Exploration stays epsilon-greedy as in the round 9 DQN (no noisy nets, no
distributional head), and every other setting matches rl.train's DQN: buffer 200k,
learning starts 20k, batch 256, lr 1e-4, train every 4 decisions, target update
every 10k, epsilon 1 -> 0.02 over 200k decisions, gamma 0.999, grad clip 10.

`learn` evaluates on an eval env every `eval_every` decisions and writes
evaluations.npz (timesteps, results, ep_lengths) the way SB3's EvalCallback does,
plus best_model.pt; `predict` mirrors SB3's (action, state) signature.
"""
import os
from collections import deque

import numpy as np
import torch
from torch import nn


class DuelingQNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, net_arch=(256, 256)):
        super().__init__()
        layers, d = [], obs_dim
        for h in net_arch:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.value = nn.Linear(d, 1)
        self.adv = nn.Linear(d, n_actions)

    def forward(self, x):
        h = self.body(x)
        a = self.adv(h)
        return self.value(h) + a - a.mean(1, keepdim=True)


class PrioritizedReplay:
    """Proportional prioritized replay over a sum tree (vectorised numpy)."""

    def __init__(self, capacity: int, obs_dim: int, alpha: float = 0.6, eps: float = 1e-6):
        self.cap = int(capacity)
        self.tree_n = 1 << int(np.ceil(np.log2(self.cap)))      # leaves, power of two
        self.tree = np.zeros(2 * self.tree_n)
        self.alpha, self.eps = alpha, eps
        self.obs = np.zeros((self.cap, obs_dim), np.float32)
        self.next_obs = np.zeros((self.cap, obs_dim), np.float32)
        self.act = np.zeros(self.cap, np.int64)
        self.ret = np.zeros(self.cap, np.float32)
        self.disc = np.zeros(self.cap, np.float32)   # gamma^n, 0 when the episode ended
        self.pos, self.size, self.max_p = 0, 0, 1.0

    def _set(self, idx: np.ndarray, p: np.ndarray) -> None:
        node = idx + self.tree_n
        self.tree[node] = p
        node = np.unique(node // 2)
        while node[0] >= 1:
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
            if node[0] == 1:
                break
            node = np.unique(node // 2)

    def add(self, o, a, r, o2, disc) -> None:
        i = self.pos
        self.obs[i], self.act[i], self.ret[i], self.next_obs[i], self.disc[i] = o, a, r, o2, disc
        self._set(np.array([i]), np.array([self.max_p]))
        self.pos = (self.pos + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch: int, beta: float, rng: np.random.Generator):
        total = self.tree[1]
        u = (np.arange(batch) + rng.random(batch)) * (total / batch)   # stratified
        node = np.ones(batch, np.int64)
        while node[0] < self.tree_n:
            left = 2 * node
            right = u > self.tree[left]
            u = np.where(right, u - self.tree[left], u)
            node = left + right
        idx = np.minimum(node - self.tree_n, self.size - 1)
        p = np.maximum(self.tree[idx + self.tree_n], self.eps) / total
        w = (self.size * p) ** -beta
        w = w / w.max()
        return idx, w.astype(np.float32)

    def update(self, idx: np.ndarray, td: np.ndarray) -> None:
        p = (np.abs(td) + self.eps) ** self.alpha
        self.max_p = max(self.max_p, float(p.max()))
        # duplicates in a batch: keep the last priority for each index
        last = {int(i): k for k, i in enumerate(idx)}
        ii = np.fromiter(last.keys(), np.int64)
        self._set(ii, p[np.fromiter(last.values(), np.int64)])


class RainbowDQN:
    def __init__(self, env, seed: int = 0, gamma: float = 0.999, n_step: int = 3,
                 buffer_size: int = 200_000, learning_starts: int = 20_000,
                 batch_size: int = 256, learning_rate: float = 1e-4, train_freq: int = 4,
                 target_update_interval: int = 10_000, exploration_steps: int = 200_000,
                 exploration_final_eps: float = 0.02, alpha: float = 0.6,
                 beta0: float = 0.4, max_grad_norm: float = 10.0, net_arch=(256, 256)):
        self.env, self.seed = env, seed
        self.gamma, self.n_step = gamma, n_step
        self.learning_starts, self.batch_size = learning_starts, batch_size
        self.train_freq, self.target_update_interval = train_freq, target_update_interval
        self.exploration_steps, self.final_eps = exploration_steps, exploration_final_eps
        self.beta0, self.max_grad_norm = beta0, max_grad_norm
        self.net_arch = tuple(net_arch)
        self.obs_dim = int(env.observation_space.shape[0])
        self.n_actions = int(env.action_space.n)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.q = DuelingQNet(self.obs_dim, self.n_actions, self.net_arch)
        self.q_target = DuelingQNet(self.obs_dim, self.n_actions, self.net_arch)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=learning_rate)
        self.buffer = PrioritizedReplay(buffer_size, self.obs_dim, alpha)
        self.num_timesteps = 0

    # ── acting ───────────────────────────────────────────────────────────────
    def _greedy(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            return int(self.q(torch.as_tensor(obs, dtype=torch.float32)[None]).argmax(1))

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        return np.int64(self._greedy(np.asarray(obs, np.float32))), None

    def epsilon(self) -> float:
        f = min(1.0, self.num_timesteps / self.exploration_steps)
        return 1.0 + f * (self.final_eps - 1.0)

    # ── learning ─────────────────────────────────────────────────────────────
    def _store(self, nq: deque, flush: bool) -> None:
        """Push n-step transitions from the queue (all of them at episode end)."""
        while len(nq) >= self.n_step or (flush and nq):
            o, a = nq[0][0], nq[0][1]
            R, g, done, o2 = 0.0, 1.0, False, nq[0][3]
            for (_, _, r, nxt, d) in list(nq)[:self.n_step]:
                R += g * r
                g *= self.gamma
                o2, done = nxt, d
                if d:
                    break
            self.buffer.add(o, a, R, o2, 0.0 if done else g)
            nq.popleft()

    def _train_step(self, total: int) -> None:
        beta = self.beta0 + (1.0 - self.beta0) * min(1.0, self.num_timesteps / total)
        idx, w = self.buffer.sample(self.batch_size, beta, self.rng)
        b = self.buffer
        o = torch.as_tensor(b.obs[idx])
        o2 = torch.as_tensor(b.next_obs[idx])
        a = torch.as_tensor(b.act[idx])
        r = torch.as_tensor(b.ret[idx])
        disc = torch.as_tensor(b.disc[idx])
        with torch.no_grad():
            a2 = self.q(o2).argmax(1, keepdim=True)                     # double Q
            target = r + disc * self.q_target(o2).gather(1, a2).squeeze(1)
        q = self.q(o).gather(1, a[:, None]).squeeze(1)
        td = target - q
        loss = (torch.as_tensor(w) * nn.functional.smooth_l1_loss(
            q, target, reduction="none")).mean()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.max_grad_norm)
        self.opt.step()
        b.update(idx, td.detach().numpy())

    def evaluate(self, eval_env, n_episodes: int):
        rewards, lengths = [], []
        for _ in range(n_episodes):
            o, _ = eval_env.reset()
            done, tot, n = False, 0.0, 0
            while not done:
                o, r, term, trunc, _ = eval_env.step(self.predict(o)[0])
                tot += r
                n += 1
                done = term or trunc
            rewards.append(tot)
            lengths.append(n)
        return rewards, lengths

    def learn(self, total_timesteps: int, eval_env=None, eval_episodes: int = 2,
              eval_every: int = 50_000, out: str | None = None, verbose: int = 1):
        evals = {"timesteps": [], "results": [], "ep_lengths": []}
        best = -np.inf
        obs, _ = self.env.reset(seed=self.seed)
        nq: deque = deque()
        while self.num_timesteps < total_timesteps:
            if self.rng.random() < self.epsilon():
                a = int(self.rng.integers(self.n_actions))
            else:
                a = self._greedy(obs)
            nxt, r, term, trunc, _ = self.env.step(a)
            done = term or trunc
            nq.append((obs, a, float(r), nxt, bool(term)))
            self._store(nq, flush=done)
            obs = nxt
            if done:
                obs, _ = self.env.reset()
            self.num_timesteps += 1
            t = self.num_timesteps
            if t > self.learning_starts and t % self.train_freq == 0:
                self._train_step(total_timesteps)
            if t % self.target_update_interval == 0:
                self.q_target.load_state_dict(self.q.state_dict())
            if eval_env is not None and t % eval_every == 0:
                rew, lens = self.evaluate(eval_env, eval_episodes)
                evals["timesteps"].append(t)
                evals["results"].append(rew)
                evals["ep_lengths"].append(lens)
                if out:
                    np.savez(os.path.join(out, "evaluations.npz"),
                             **{k: np.asarray(v) for k, v in evals.items()})
                    if np.mean(rew) > best:
                        best = float(np.mean(rew))
                        self.save(os.path.join(out, "best_model.pt"))
                if verbose:
                    print(f"Eval num_timesteps={t}, episode_reward={np.mean(rew):.2f} "
                          f"+/- {np.std(rew):.2f}", flush=True)
        return self

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        torch.save({"state_dict": self.q.state_dict(), "obs_dim": self.obs_dim,
                    "n_actions": self.n_actions, "net_arch": self.net_arch}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "RainbowPolicy":
        d = torch.load(path, map_location=device)
        return RainbowPolicy(d)


class RainbowPolicy:
    """A loaded Rainbow-style network for evaluation (predict only)."""

    def __init__(self, d: dict):
        self.q = DuelingQNet(d["obs_dim"], d["n_actions"], d["net_arch"])
        self.q.load_state_dict(d["state_dict"])
        self.q.eval()

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        with torch.no_grad():
            a = self.q(torch.as_tensor(np.asarray(obs, np.float32))[None]).argmax(1)
        return np.int64(a.item()), None
