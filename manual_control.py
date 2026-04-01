"""
Manual Control with AI Suggestions
====================================
Control the robot with keyboard while a trained DQN model
suggests the best action at every step.

Keys:
    D / A   — Move +X / -X (Right / Left)
    W / S   — Move +Y / -Y (Approach / Recede)
    R / F   — Move +Z / -Z (Rise / Lower)
    E / Q   — Rotate CW / CCW (Yaw around Z)
    SPACE   — Execute the AI-suggested action
    0       — Reset environment (random pose)
    ESC     — Quit

Usage:
    python3 manual_control.py                          # uses best model
    python3 manual_control.py --model models/final_dqn.pth
    python3 manual_control.py --random-start           # start from random pose
"""

import os
import sys
import argparse

import cv2
import numpy as np
import torch

from robot_env import RobotWallEnv

# ── action names & key map ────────────────────────────────────

ACTION_NAMES = [
    "Move +X  (D)",
    "Move -X  (A)",
    "Move +Y  (W)",
    "Move -Y  (S)",
    "Move +Z  (R)",
    "Move -Z  (F)",
    "Rotate CW  (E)",
    "Rotate CCW (Q)",
]

ACTION_SHORT = [
    "→ +X", "← -X", "↑ +Y", "↓ -Y",
    "⬆ +Z", "⬇ -Z", "↻ CW", "↺ CCW",
]

KEY_TO_ACTION = {
    ord('d'): 0,   # +X
    ord('a'): 1,   # -X
    ord('w'): 2,   # +Y
    ord('s'): 3,   # -Y
    ord('r'): 4,   # +Z
    ord('f'): 5,   # -Z
    ord('e'): 6,   # CW
    ord('q'): 7,   # CCW
}


# ── load trained model ────────────────────────────────────────

def load_model(model_path, device):
    """Load a trained QNetwork from checkpoint."""
    # import here so script works even if torch isn't in path
    sys.path.insert(0, os.path.dirname(__file__))
    from train import QNetwork

    net = QNetwork().to(device)
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        net.load_state_dict(ckpt["q_net"])
        net.eval()
        print(f"  ✓ Loaded model: {model_path}")
        return net
    else:
        print(f"  ✗ Model not found: {model_path}")
        print("    Running without AI suggestions.")
        return None


def get_suggestion(net, state, device):
    """Get the model's suggested action and Q-values."""
    with torch.no_grad():
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = net(s).cpu().numpy()[0]
    best = int(np.argmax(q_values))
    return best, q_values


# ── overlay HUD on the frame ─────────────────────────────────

def draw_hud(frame, env, obs, reward, suggestion, q_values, step_num):
    """Draw pose info, reward, and AI suggestion on the frame."""
    H, W = frame.shape[:2]

    # ── semi-transparent panel on the right ───────────────────
    panel_w = 320
    overlay = frame.copy()
    cv2.rectangle(overlay, (W - panel_w, 0), (W, H),
                  (30, 30, 30), cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    x0 = W - panel_w + 15
    y  = 30
    white  = (255, 255, 255)
    grey   = (160, 160, 160)
    green  = (0, 255, 100)
    yellow = (0, 230, 255)
    cyan   = (255, 230, 0)
    font   = cv2.FONT_HERSHEY_SIMPLEX

    # ── title ─────────────────────────────────────────────────
    cv2.putText(frame, "MANUAL CONTROL", (x0, y),
                font, 0.65, cyan, 2)
    y += 30

    # ── step & reward ─────────────────────────────────────────
    cv2.putText(frame, f"Step: {step_num}", (x0, y),
                font, 0.5, grey, 1)
    y += 22
    cv2.putText(frame, f"Reward: {reward:+.4f}", (x0, y),
                font, 0.5, white, 1)
    y += 22

    # ── pose ──────────────────────────────────────────────────
    cv2.putText(frame,
                f"X={env.rx:+.2f}  Y={env.ry:+.2f}  "
                f"Z={env.rz:+.2f}", (x0, y),
                font, 0.45, grey, 1)
    y += 20
    cv2.putText(frame, f"Yaw={env.yaw:+.1f} deg", (x0, y),
                font, 0.45, grey, 1)
    y += 30

    # ── visibility ────────────────────────────────────────────
    vis = obs["marker_visible"]
    num_vis = int(vis.sum())
    cv2.putText(frame, f"Markers visible: {num_vis}/8", (x0, y),
                font, 0.5, green if num_vis == 8 else yellow, 1)
    y += 22

    # ── pixel error display ──────────────────────────────
    total_error = getattr(env, '_total_error', 0.0)
    err_col = green if total_error < 24.0 else yellow  # 3px × 8 markers
    cv2.putText(frame, f"Total Error: {total_error:.1f}px", (x0, y),
                font, 0.5, err_col, 1)
    y += 22

    # ── marker areas ─────────────────────────────────────
    areas = obs.get("marker_areas", np.zeros(8))
    mean_area = float(np.mean(areas[vis.astype(bool)])) if num_vis > 0 else 0.0
    cv2.putText(frame, f"Mean Area: {mean_area:.6f}", (x0, y),
                font, 0.5, grey, 1)
    y += 30

    # ── AI suggestion ─────────────────────────────────────────
    if suggestion is not None:
        cv2.putText(frame, "AI SUGGESTS:", (x0, y),
                    font, 0.55, cyan, 2)
        y += 28
        cv2.putText(frame, f"  >> {ACTION_SHORT[suggestion]} <<", (x0, y),
                    font, 0.7, green, 2)
        y += 28
        cv2.putText(frame, "(press SPACE to execute)", (x0, y),
                    font, 0.4, grey, 1)
        y += 30

        # ── Q-value bar chart ─────────────────────────────────
        cv2.putText(frame, "Q-Values:", (x0, y),
                    font, 0.45, white, 1)
        y += 20

        q_min = q_values.min()
        q_max = q_values.max()
        q_range = max(q_max - q_min, 1e-6)

        for i in range(8):
            bar_len = int(120 * (q_values[i] - q_min) / q_range)
            colour = green if i == suggestion else grey
            label = f"{ACTION_SHORT[i]:>7}"
            cv2.putText(frame, label, (x0, y),
                        font, 0.35, colour, 1)
            cv2.rectangle(frame,
                          (x0 + 75, y - 8),
                          (x0 + 75 + bar_len, y + 2),
                          colour, cv2.FILLED)
            cv2.putText(frame, f"{q_values[i]:+.2f}",
                        (x0 + 200, y), font, 0.3, grey, 1)
            y += 18
    else:
        cv2.putText(frame, "No model loaded", (x0, y),
                    font, 0.5, (0, 0, 200), 1)
        y += 25

    y += 15

    # ── key legend ────────────────────────────────────────────
    cv2.putText(frame, "KEYS:", (x0, y), font, 0.45, white, 1)
    y += 20
    for line in ["D/A  +X/-X (Right/Left)", "W/S  +Y/-Y (Forward/Backward)",
                 "R/F  +Z/-Z (Up/Down)", "E/Q  Rotate CW/CCW (Yaw)",
                 "SPC  Use AI action", "0    Reset", "ESC  Quit"]:
        cv2.putText(frame, line, (x0, y), font, 0.35, grey, 1)
        y += 16

    # ── reference boxes (red) on the main view ────────────────
    for i in range(8):
        bx1, by1, bx2, by2 = env.ref_boxes_px[i].astype(int)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 180), 1)

    # ── error bar at bottom-left ──────────────────────────
    bar_y = H - 60
    cv2.putText(frame, "Error:", (20, bar_y - 5),
                font, 0.4, grey, 1)
    total_err = getattr(env, '_total_error', 0.0)
    # normalise: 0 error = full bar, max ~1280px error = empty bar
    max_err = np.sqrt(W**2 + H**2)
    err_frac = max(0.0, 1.0 - total_err / max_err)
    bar_w = int(250 * err_frac)
    bar_col = green if getattr(env, '_aligned', False) else yellow
    cv2.rectangle(frame, (110, bar_y - 12), (110 + 250, bar_y + 4),
                  (60, 60, 60), cv2.FILLED)
    cv2.rectangle(frame, (110, bar_y - 12), (110 + bar_w, bar_y + 4),
                  bar_col, cv2.FILLED)
    cv2.putText(frame, f"{total_err:.1f}px", (370, bar_y),
                font, 0.4, white, 1)

    # ── per-marker pixel dist row at very bottom ─────────
    bar_y2 = H - 25
    pdists = getattr(env, '_pixel_dists', np.zeros(8))
    dist_str = "  ".join(f"{pdists[i]:.1f}" if vis[i] else " -- " for i in range(8))
    cv2.putText(frame, f"Px: {dist_str}", (20, bar_y2),
                font, 0.35, grey, 1)

    return frame


# ── main loop ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Manual control with AI suggestions")
    parser.add_argument("--model", type=str,
                        default="models/best_dqn.pth",
                        help="Path to trained DQN checkpoint")
    parser.add_argument("--random-start", action="store_true",
                        help="Start from a random pose")
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("\n" + "=" * 56)
    print("  Manual Control with AI Suggestions")
    print("=" * 56)

    env = RobotWallEnv()
    env.random_reset = args.random_start

    net = load_model(args.model, device)

    obs, _ = env.reset()
    state  = RobotWallEnv.flatten_obs(obs)
    reward = env._compute_reward(obs, action=-1)
    step_num = 0

    print("\n  Ready! Use keyboard to control the robot.")
    print("  The AI suggestion is shown on screen.\n")

    while True:
        # ── get AI suggestion ─────────────────────────────────
        suggestion, q_values = (None, None)
        if net is not None:
            suggestion, q_values = get_suggestion(net, state, device)

        # ── render frame with HUD ─────────────────────────────
        frame, corners, ids = env._render_frame()

        # draw detected markers
        if corners is not None:
            for i, c in enumerate(corners):
                pts = c[0].astype(int)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                if ids is not None:
                    mid = int(ids[i][0])
                    cx = int(pts[:, 0].mean())
                    cy = int(pts[:, 1].mean())
                    cv2.putText(frame, f"A:{mid}",
                                (cx - 20, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

        frame = draw_hud(frame, env, obs, reward,
                         suggestion, q_values, step_num)

        cv2.imshow("Manual Control + AI Suggestions", frame)
        key = cv2.waitKeyEx(0)

        # ── handle input ──────────────────────────────────────
        if key == 27:  # ESC
            break

        elif key == ord('0'):  # reset
            obs, _ = env.reset()
            state  = RobotWallEnv.flatten_obs(obs)
            reward = env._compute_reward(obs, action=-1)
            step_num = 0
            print("  ↻ Environment reset")
            continue

        elif key == 32 and suggestion is not None:  # SPACE
            action = suggestion
            print(f"  [AI] Step {step_num}: "
                  f"{ACTION_NAMES[action]}  "
                  f"(Q={q_values[action]:+.3f})")

        elif key in KEY_TO_ACTION:
            action = KEY_TO_ACTION[key]
            ai_tag = ""
            if suggestion is not None:
                if action == suggestion:
                    ai_tag = "  ✓ matches AI"
                else:
                    ai_tag = (f"  (AI preferred: "
                              f"{ACTION_SHORT[suggestion]})")
            print(f"  [YOU] Step {step_num}: "
                  f"{ACTION_NAMES[action]}{ai_tag}")

        else:
            continue

        # ── execute action ────────────────────────────────────
        obs, reward, terminated, truncated, _ = env.step(action)
        state = RobotWallEnv.flatten_obs(obs)
        step_num += 1

        if terminated:
            print(f"\n  ★ ALIGNED at step {step_num}! "
                  f"Reward: {reward:.4f}")
            print("    Press 0 to reset or ESC to quit.\n")

        if truncated:
            print(f"\n  ⏰ Truncated at {step_num} steps. "
                  f"Press 0 to reset.\n")

    cv2.destroyAllWindows()
    print("\n  Goodbye!\n")


if __name__ == "__main__":
    main()
