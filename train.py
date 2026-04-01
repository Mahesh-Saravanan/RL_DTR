"""
Pure PyTorch DQN Training for RobotWallEnv
===========================================
Trains an agent to align a mecanum robot with a marker wall
using a from-scratch Double DQN implementation.

DQN Config: 8-Marker Alignment (XY = Ground)
  State:  24-dim (16 pixel offsets + 8 marker areas)
  Action: 8 discrete (+/-X, +/-Y, +/-Z, CW/CCW)
  Reward: Potential-based + terminal + safety + step cost

Usage:
    python3 train.py                   # train 10000 episodes
    python3 train.py --episodes 5000   # train fewer
"""

import os
import sys
import csv
import time
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from robot_env import RobotWallEnv


# ═══════════════════════════════════════════════════════════════
#  Hyper-parameters
# ═══════════════════════════════════════════════════════════════

STATE_DIM       = 24        # 16 pixel offsets + 8 marker areas
ACTION_DIM      = 8
HIDDEN          = 128       # hidden layer size
LR              = 0.00025   # stable rate for 24-dim input
GAMMA           = 0.97      # values final docking, prioritises immediate correction
BATCH_SIZE      = 128
BUFFER_SIZE     = 500_000
EPS_START       = 1.0
EPS_END         = 0.05
# Linear decay: eps decreases from 1.0 to 0.05 over EPS_DECAY_STEPS
EPS_DECAY_STEPS = 500_000   # steps over which epsilon decays linearly
TAU             = 0.005     # soft target update blending factor
NUM_EPISODES    = 10000
MAX_STEPS       = 800
MODEL_DIR       = "models"
LOG_DIR         = "logs"
BUFFER_WARMUP   = 5000      # transitions before first gradient step

# ═══════════════════════════════════════════════════════════════
#  Replay Buffer
# ═══════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity, state_dim=STATE_DIM):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        idx = np.random.randint(0, self.size, size=n)

        states = torch.FloatTensor(self.states[idx])
        actions = torch.LongTensor(self.actions[idx]).squeeze(1)
        rewards = torch.FloatTensor(self.rewards[idx]).squeeze(1)
        next_states = torch.FloatTensor(self.next_states[idx])
        dones = torch.FloatTensor(self.dones[idx]).squeeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size


# ═══════════════════════════════════════════════════════════════
#  Q-Network  (MLP: 24 → 128 → 128 → 8)  [offsets+areas → actions]
# ═══════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════════
#  DQN Agent
# ═══════════════════════════════════════════════════════════════

class DQNAgent:
    def __init__(self, device=None):
        self.device = device or torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.q_net      = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.replay    = ReplayBuffer(BUFFER_SIZE)

    # ── action selection ─────────────────────────────────────
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(ACTION_DIM)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_net(s).argmax(dim=1).item()

    # ── single gradient step (Double-DQN) ────────────────────
    def train_step(self, total_steps=0):
        if len(self.replay) < BATCH_SIZE or total_steps < BUFFER_WARMUP:
            return 0.0

        states, actions, rewards, next_states, dones = \
            self.replay.sample(BATCH_SIZE)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # current Q(s, a)
        q_values = self.q_net(states) \
                       .gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            best_actions = self.q_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states) \
                         .gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target = rewards + GAMMA * next_q * (1.0 - dones)

        loss = nn.SmoothL1Loss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # ── Polyak Averaging (Soft Target Update) ────────────────
        for target_param, q_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(TAU * q_param.data + (1.0 - TAU) * target_param.data)

        return loss.item()

    # ── hard target update ───────────────────────────────────
    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ── save / load ──────────────────────────────────────────
    def save(self, path):
        torch.save({
            "q_net":      self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])


# ═══════════════════════════════════════════════════════════════
#  Convergence plots
# ═══════════════════════════════════════════════════════════════

def plot_training_curves(csv_path):
    """Read logs/training_log.csv and produce convergence plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")                         # no GUI needed
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [!] matplotlib not installed — skipping plots.")
        print("      Install with:  pip install matplotlib")
        return

    # ── read CSV ──────────────────────────────────────────────
    eps, rewards, avg_rewards = [], [], []
    succ_rates, losses, epsilons = [], [], []
    steps_list = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eps.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            avg_rewards.append(float(row["avg_reward_100"]))
            succ_rates.append(float(row["success_rate_100"]))
            losses.append(float(row["avg_loss"]))
            epsilons.append(float(row["epsilon"]))
            steps_list.append(int(row["steps"]))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("DQN Training Convergence", fontsize=15, fontweight="bold")

    # 1 — Episode reward + rolling average
    ax = axes[0, 0]
    ax.plot(eps, rewards, alpha=0.25, color="steelblue", label="Episode")
    ax.plot(eps, avg_rewards, color="navy", linewidth=2, label="Avg-100")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Episode Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2 — Success rate
    ax = axes[0, 1]
    ax.plot(eps, [s * 100 for s in succ_rates], color="seagreen", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success %")
    ax.set_title("Alignment Success Rate (100-ep window)")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    # 3 — Average loss
    ax = axes[1, 0]
    ax.plot(eps, losses, color="firebrick", alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Loss")
    ax.set_title("Training Loss (per step)")
    ax.grid(True, alpha=0.3)

    # 4 — Epsilon + steps per episode
    ax = axes[1, 1]
    ax2 = ax.twinx()
    l1, = ax.plot(eps, epsilons, color="darkorange", linewidth=2,
                  label="Epsilon")
    l2, = ax2.plot(eps, steps_list, color="purple", alpha=0.4,
                   label="Steps/ep")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon", color="darkorange")
    ax2.set_ylabel("Steps per episode", color="purple")
    ax.set_title("Exploration & Episode Length")
    ax.legend(handles=[l1, l2], loc="center right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(LOG_DIR, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  ✓ Convergence plots saved to: {plot_path}")


# ═══════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════

def train(num_episodes=NUM_EPISODES, max_steps=MAX_STEPS,
          visualize_eps=0):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    env = RobotWallEnv()
    env.random_reset = True
    env.max_steps    = max_steps

    agent   = DQNAgent()
    epsilon = EPS_START

    best_avg     = -float("inf")
    reward_hist  = deque(maxlen=100)
    success_hist = deque(maxlen=100)
    total_steps  = 0

    # ── CSV logger ───────────────────────────────────────────
    csv_path = os.path.join(LOG_DIR, "training_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "episode", "steps", "reward", "avg_reward_100",
        "success_rate_100", "epsilon", "avg_loss", "aligned",
    ])

    print(f"\n{'='*64}")
    print(f"  DQN Training  —  8-Marker Alignment (XY=Ground)")
    print(f"  State:  {STATE_DIM} (16 offsets + 8 areas)  Actions: {ACTION_DIM}")
    print(f"  γ={GAMMA}  α={LR}  ε: {EPS_START}→{EPS_END} (linear/{EPS_DECAY_STEPS} steps)")
    print(f"  Device: {agent.device}")
    print(f"  Episodes: {num_episodes}   Max steps/ep: {max_steps}")
    print(f"  CSV log: {csv_path}")
    print(f"{'='*64}\n")
    print(f"{'Ep':>6} {'Steps':>6} {'Reward':>8} {'Avg100':>8} "
          f"{'Succ%':>6} {'Eps':>6} {'Loss':>8}  Status")
    print("-" * 64)

    t0 = time.time()

    for ep in range(1, num_episodes + 1):
        obs, _  = env.reset()
        state   = RobotWallEnv.flatten_obs(obs)
        ep_reward = 0.0
        ep_loss   = 0.0
        steps     = 0
        aligned   = False

        for _ in range(max_steps):
            action = agent.select_action(state, epsilon)
            obs, reward, terminated, truncated, _ = env.step(action)
            next_state = RobotWallEnv.flatten_obs(obs)
            done = terminated or truncated

            agent.replay.push(state, action, reward,
                              next_state, float(done))
            ep_loss += agent.train_step(total_steps)

            # live visualisation for first N episodes
            if ep <= visualize_eps:
                action_names = [
                    "+X Right", "-X Left", "+Y Approach", "-Y Recede",
                    "+Z Rise", "-Z Lower", "Rot CW", "Rot CCW",
                ]
                env._current_action_name = action_names[action]
                env.visualize(reward)
                import cv2
                cv2.waitKey(50)

            state      = next_state
            ep_reward += reward
            steps     += 1
            total_steps += 1

            # ── linear epsilon decay per-step ────────────────────────
            epsilon = max(EPS_END,
                          EPS_START - (EPS_START - EPS_END) * total_steps / EPS_DECAY_STEPS)

            if done:
                aligned = terminated
                break

        if ep == visualize_eps:
            import cv2
            cv2.destroyAllWindows()

        reward_hist.append(ep_reward)
        success_hist.append(1.0 if aligned else 0.0)

        avg  = np.mean(reward_hist)
        succ = np.mean(success_hist)

        if avg > best_avg and len(reward_hist) >= 50:
            best_avg = avg
            agent.save(os.path.join(MODEL_DIR, "best_dqn.pth"))

        # ── write CSV row (every episode) ────────────────────
        csv_writer.writerow([
            ep, steps, f"{ep_reward:.4f}", f"{avg:.4f}",
            f"{succ:.4f}", f"{epsilon:.4f}",
            f"{ep_loss/max(1,steps):.6f}", int(aligned),
        ])
        csv_file.flush()

        # ── checkpoint every 200 episodes (overwrite) ────────
        if ep % 200 == 0:
            ckpt_path = os.path.join(MODEL_DIR, "checkpoint.pth")
            agent.save(ckpt_path)
            print(f"       ↳ Checkpoint saved → {ckpt_path}")

        # ── early stop if learned enough ─────────────────────
        if len(success_hist) >= 100 and succ >= 0.90:
            print(f"\n  ★ Early stop at ep {ep}: "
                  f"success rate {succ*100:.1f}% ≥ 90% over 100 eps")
            break

        # ── terminal log (periodic) ──────────────────────────
        if ep % 25 == 0 or ep == 1:
            tag = "ALIGNED" if aligned else ""
            print(f"{ep:>6} {steps:>6} {ep_reward:>8.2f} {avg:>8.2f} "
                  f"{succ*100:>5.1f}% {epsilon:>6.3f} "
                  f"{ep_loss/max(1,steps):>8.4f}  {tag}")

    csv_file.close()

    # ── save final model ─────────────────────────────────────
    final_path = os.path.join(MODEL_DIR, "final_dqn.pth")
    agent.save(final_path)

    elapsed = time.time() - t0
    print(f"\n{'='*64}")
    print(f"  Training complete in {elapsed:.1f}s")
    print(f"  Total steps:  {total_steps:,}")
    print(f"  Best avg-100: {best_avg:.2f}")
    print(f"  Final model:  {final_path}")
    print(f"  Best model:   {os.path.join(MODEL_DIR, 'best_dqn.pth')}")
    print(f"  CSV log:      {csv_path}")
    print(f"{'='*64}\n")

    # ── generate convergence plots ───────────────────────────
    plot_training_curves(csv_path)


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=NUM_EPISODES)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--visualize", type=int, default=0, metavar="N",
                   help="Show live camera view for the first N episodes")
    args = p.parse_args()
    train(num_episodes=args.episodes, max_steps=args.max_steps,
          visualize_eps=args.visualize)
