"""
Evaluate a trained DQN model on RobotWallEnv
=============================================
Loads a saved model, runs episodes with random initial poses,
and visualises the agent's decisions in real time.

Usage:
    python3 eval.py                          # best model, 5 eps
    python3 eval.py models/final_dqn.pth 10  # specific model, 10 eps
"""

import argparse
import sys
import cv2
import numpy as np
import torch

from robot_env import RobotWallEnv
from train import QNetwork, STATE_DIM, ACTION_DIM


ACTION_NAMES = [
    "+X Right", "-X Left", "+Y Approach", "-Y Recede",
    "+Z Rise", "-Z Lower", "Rot CW", "Rot CCW",
]


def load_model(path):
    """Load a saved QNetwork from a checkpoint file."""
    device = torch.device("cpu")
    model  = QNetwork(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    ckpt   = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["q_net"])
    model.eval()
    return model, device


def evaluate(model_path="models/best_dqn.pth", episodes=5, level=3):
    """Run evaluation episodes with live visualisation."""
    model, device = load_model(model_path)

    env = RobotWallEnv()
    env.random_reset = True
    
    # ── Set Curriculum Level ─────────────────────────────
    if level == 1:
        env.reset_x_range   = 0.1
        env.reset_z_range   = 0.1
        env.reset_y_range   = 0.1
        env.reset_yaw_range = 5.0
    elif level == 2:
        env.reset_x_range   = 0.3
        env.reset_z_range   = 0.2
        env.reset_y_range   = 0.3
        env.reset_yaw_range = 25.0
    elif level == 3:
        env.reset_x_range   = 0.5
        env.reset_z_range   = 0.3
        env.reset_y_range   = 0.5
        env.reset_yaw_range = 40.0
    else:  # Level 4 (Extreme)
        env.reset_x_range   = 0.8
        env.reset_z_range   = 0.4
        env.reset_y_range   = 1.0
        env.reset_yaw_range = 80.0

    print(f"\n{'='*60}")
    print(f"  DQN Evaluation  —  {model_path}")
    print(f"  Episodes: {episodes}   |   Level: {level}")
    print(f"{'='*60}")

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        state  = RobotWallEnv.flatten_obs(obs)
        total_reward = 0.0

        print(f"\n── Episode {ep} ──")
        spec_z = env.ry
        spec_y = -env.rz
        print(f"   Start  X={env.rx:+.2f}  Y={spec_y:+.2f}  "
              f"Z={spec_z:+.2f}  Yaw={env.yaw:+.1f}°")

        for step in range(env.max_steps):
            # get Q-values and pick best action
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(device)
                q = model(s).squeeze().cpu().numpy()

            action = int(q.argmax())

            # show the chosen action and top-3 Q-values
            sorted_idx = np.argsort(q)[::-1]
            top3 = "  ".join(
                f"{ACTION_NAMES[i]}={q[i]:+.2f}" for i in sorted_idx[:3]
            )
            print(f"   Step {step+1:>3}: {ACTION_NAMES[action]:>10}  │ Q: {top3}")

            obs, reward, terminated, truncated, _ = env.step(action)
            state = RobotWallEnv.flatten_obs(obs)
            total_reward += reward

            env._current_action_name = ACTION_NAMES[action]
            env.visualize(reward)
            key = cv2.waitKey(80)      # ~12 fps playback
            if key == 27:
                cv2.destroyAllWindows()
                return

            if terminated:
                print(f"   ✓ ALIGNED in {step+1} steps  │  "
                      f"Total reward: {total_reward:.2f}  │  "
                      f"Final Error: {getattr(env, '_total_error', 0.0):.1f}px")
                print("   [Press SPACE to continue, ESC to exit]")
                while True:
                    k = cv2.waitKey(0)
                    if k == 32:  # Space
                        break
                    elif k == 27:  # ESC
                        cv2.destroyAllWindows()
                        return
                break
            if truncated:
                print(f"   ✗ TIMEOUT after {step+1} steps  │  "
                      f"Total reward: {total_reward:.2f}  │  "
                      f"Final Error: {getattr(env, '_total_error', 0.0):.1f}px")
                break

        # pause between episodes
        cv2.waitKey(1500)

    cv2.destroyAllWindows()
    print(f"\nDone — {episodes} episodes evaluated.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQN Model")
    parser.add_argument("model", type=str, nargs="?", default="models/best_dqn.pth",
                        help="Path to the model to evaluate")
    parser.add_argument("--eps", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--level", type=int, choices=[1, 2, 3, 4], default=4,
                        help="Curriculum level (1=easy, 2=med, 3=hard, 4=extreme)")
    
    args = parser.parse_args()
    evaluate(args.model, args.eps, args.level)
