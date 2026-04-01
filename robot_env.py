import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RobotWallEnv(gym.Env):
    """
    Simulates a mecanum-wheel mobile robot with a forward-facing camera
    viewing a flat white wall with 8 ArUco markers (4x2 grid).

    Coordinate System (XY = Ground):
        X-axis: Lateral movement (Left/Right) along the floor.
        Y-axis: Depth movement (Toward/Away from the wall).
        Z-axis: Vertical movement (Up/Down).
        Rotation: Yaw (Rotation around the Z-axis).

    State Space (24 Dimensions):
        Pixel Offsets (16 values): [x_1, y_1, ... x_8, y_8]
            Normalized pixel distance from each marker center to its
            corresponding target box center.
        Marker Areas (8 values): Normalized pixel area of each marker.
            Critical signal for Y-axis depth; as robot moves in +Y
            these values increase.

    Actions (Discrete 8):
        0: Move +X   (Slide Right)
        1: Move -X   (Slide Left)
        2: Move +Y   (Approach wall)
        3: Move -Y   (Recede from wall)
        4: Move +Z   (Rise)
        5: Move -Z   (Lower)
        6: Rotate CW  (Yaw around Z)
        7: Rotate CCW (Yaw around Z)

    Reward Function:
        Potential-Based: 10.0 × (E_{t-1} - E_t)
        Terminal Success: +500 if E < threshold (all markers within 3px)
        Safety Penalty:  -100 if any marker lost or robot out of XY bounds
        Step Cost:       -0.5 per action
    """

    def __init__(self):

        # ── image dimensions ──────────────────────────────────
        self.W = 1024
        self.H = 768

        # ── wall physical size (metres) ──────────────────────
        self.wall_width  = 0.95
        self.wall_height = 1.60

        # ── marker grid ──────────────────────────────────────
        self.cols = 2
        self.rows = 4
        self.marker_ratio = 0.30          # marker size relative to cell

        self.cell_w = self.wall_width  / self.cols
        self.cell_h = self.wall_height / self.rows
        self.marker_size = min(self.cell_w, self.cell_h) * self.marker_ratio

        # ── camera intrinsics ────────────────────────────────
        self.hfov = 60.0                  # horizontal field-of-view (deg)
        self.fx = self.W / (2.0 * np.tan(np.radians(self.hfov / 2.0)))
        self.fy = self.fx
        self.cx = self.W  / 2.0
        self.cy = self.H  / 2.0

        self.K = np.array([
            [self.fx,  0,      self.cx],
            [0,        self.fy, self.cy],
            [0,        0,      1      ],
        ], dtype=np.float64)

        # ── ArUco setup ──────────────────────────────────────
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict,
            cv2.aruco.DetectorParameters(),
        )

        # ── movement parameters ──────────────────────────────
        self.move_step   = 0.003          # metres per action (~1.3 px at 2m)
        self.rotate_step = 0.2            # degrees per action (~3.1 px at 2m)

        # ── starting pose ────────────────────────────────────
        # Internal: rx=X(lateral), ry=Y_up(vertical), rz=Z_fwd(depth/negative)
        # Mapping to spec: spec_X=rx, spec_Y=-rz, spec_Z=ry
        self.start_distance = 2.0        # metres in front of wall

        # ── training parameters ──────────────────────────────
        self.max_steps    = 800           # steps before truncation
        self.align_pixel_thresh = 3.0     # all markers within 5px → success
        self.random_reset = False         # set True for training
        self._prev_error  = 0.0           # for potential-based reward
        self._aligned = False             # strict alignment flag
        self.current_step = 0

        # ── XY boundary limits (spec X, spec Y) ─────────────
        self.x_bound = (-1.0, 1.0)        # lateral bounds (metres)
        self.y_bound = (0.5, 3.5)          # depth bounds (distance from wall)

        # ── area normalisation ───────────────────────────────
        self.max_marker_area = float(self.W * self.H)  # theoretical max

        # ── spaces ───────────────────────────────────────────
        self.action_space = spaces.Discrete(8)

        # 24-dim: 16 pixel offsets + 8 marker areas
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32
        )

        # ── pre-render wall texture ──────────────────────────
        self.wall_ppm = 600               # texture pixels-per-metre
        self._create_wall_texture()

        # ── compute reference positions at starting pose ─────
        self.rx  = 0.0                    # X lateral
        self.ry  = 0.0                    # Y_up (vertical / spec Z)
        self.rz  = -self.start_distance   # Z_fwd (depth / spec -Y)
        self.yaw = 0.0
        self._compute_reference_positions()

        self.reset()

    # ==========================================================
    #  Wall texture
    # ==========================================================

    def _create_wall_texture(self):
        """Pre-render a white wall image with ArUco markers baked in."""
        tw = int(self.wall_width  * self.wall_ppm)
        th = int(self.wall_height * self.wall_ppm)

        self.wall_texture = np.ones((th, tw, 3), dtype=np.uint8) * 255

        cell_pw = tw // self.cols
        cell_ph = th // self.rows
        marker_px = int(min(cell_pw, cell_ph) * self.marker_ratio)
        if marker_px % 2 == 1:
            marker_px += 1                # keep even for centering

        mid = 0
        for r in range(self.rows):
            for c in range(self.cols):
                ccx = c * cell_pw + cell_pw // 2
                ccy = r * cell_ph + cell_ph // 2

                marker_img = cv2.aruco.generateImageMarker(
                    self.aruco_dict, mid, marker_px
                )
                marker_img = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

                mx = ccx - marker_px // 2
                my = ccy - marker_px // 2
                self.wall_texture[my:my + marker_px,
                                  mx:mx + marker_px] = marker_img
                mid += 1

    # ==========================================================
    #  3-D helpers
    # ==========================================================

    def _rotation_matrix(self):
        """World-to-camera rotation for current yaw.

        Yaw is rotation around the Z-axis (vertical in spec,
        Y-up in internal world frame).

        Camera convention (OpenCV):  x-right, y-down, z-forward.
        World convention:            X-right, Y-up,   Z-forward.
        """
        t = np.radians(self.yaw)
        ct, st = np.cos(t), np.sin(t)
        # Yaw around world Y-up axis (same as spec Z-vertical)
        return np.array([
            [ ct,  0, -st],
            [  0, -1,   0],
            [ st,  0,  ct],
        ], dtype=np.float64)

    def _camera_Rt(self):
        """Return (R, t) such that  P_cam = R @ P_world + t."""
        R = self._rotation_matrix()
        t = -R @ np.array([self.rx, self.ry, self.rz])
        return R, t

    def _wall_homography(self):
        """Homography that maps wall-texture pixels → image pixels.

        Uses the plane-at-Z=0 shortcut:
            H_plane = K @ [r1 | r2 | t]
        then chains with texture→physical mapping.
        """
        R, t = self._camera_Rt()
        H_plane = self.K @ np.column_stack([R[:, 0], R[:, 1], t])

        # texture-pixel (u,v) → physical (X,Y) on the wall
        ppm = self.wall_ppm
        hw  = self.wall_width  / 2.0
        hh  = self.wall_height / 2.0
        M_tex = np.array([
            [1.0 / ppm,  0,          -hw],
            [0,         -1.0 / ppm,   hh],
            [0,          0,            1 ],
        ], dtype=np.float64)

        return H_plane @ M_tex

    def _project_points(self, pts_3d):
        """Project Nx3 world points → Nx2 image pixels.
        Returns (pts_2d, valid_mask).
        """
        R, t = self._camera_Rt()
        cam = (R @ pts_3d.T).T + t        # Nx3

        valid = cam[:, 2] > 0.01
        pts_2d = np.full((len(pts_3d), 2), -1e4, dtype=np.float64)
        if np.any(valid):
            z = cam[valid, 2]
            pts_2d[valid, 0] = self.fx * cam[valid, 0] / z + self.cx
            pts_2d[valid, 1] = self.fy * cam[valid, 1] / z + self.cy
        return pts_2d, valid

    def _wall_visible(self):
        """Quick check: is the wall centre in front of the camera?"""
        R, t = self._camera_Rt()
        centre_cam = R @ np.array([0.0, 0.0, 0.0]) + t
        return centre_cam[2] > 0.01

    # ==========================================================
    #  Reference positions (fixed in camera frame)
    # ==========================================================

    def _marker_3d_centres_and_boxes(self):
        """Return lists of (centre_3d, corners_3d) for each marker."""
        hw = self.wall_width  / 2.0
        hh = self.wall_height / 2.0
        ms = self.marker_size / 2.0
        markers = []
        for r in range(self.rows):
            for c in range(self.cols):
                cx = -hw + (c + 0.5) * self.cell_w
                cy =  hh - (r + 0.5) * self.cell_h
                centre  = np.array([[cx, cy, 0.0]])
                corners = np.array([
                    [cx - ms, cy + ms, 0.0],
                    [cx + ms, cy + ms, 0.0],
                    [cx + ms, cy - ms, 0.0],
                    [cx - ms, cy - ms, 0.0],
                ])
                markers.append((centre, corners))
        return markers

    def _compute_reference_positions(self):
        """Compute and store reference marker image positions at current pose."""
        markers = self._marker_3d_centres_and_boxes()
        self.ref_centers_px = np.zeros((8, 2), dtype=np.float32)
        self.ref_boxes_px   = np.zeros((8, 4), dtype=np.float32)
        self.ref_areas_px   = np.zeros(8, dtype=np.float32)
        for i, (ctr3, cor3) in enumerate(markers):
            c2, _ = self._project_points(ctr3)
            b2, _ = self._project_points(cor3)
            self.ref_centers_px[i] = c2[0]
            x1, y1 = b2[:, 0].min(), b2[:, 1].min()
            x2, y2 = b2[:, 0].max(), b2[:, 1].max()
            self.ref_boxes_px[i] = [x1, y1, x2, y2]
            self.ref_areas_px[i] = (x2 - x1) * (y2 - y1)

    # ==========================================================
    #  Gym interface
    # ==========================================================

    def _spec_y(self):
        """Current depth (spec Y) = distance from wall = -rz."""
        return -self.rz

    def _out_of_bounds(self):
        """Check if robot has exceeded XY boundary limits."""
        spec_x = self.rx
        spec_y = self._spec_y()
        return (spec_x < self.x_bound[0] or spec_x > self.x_bound[1] or
                spec_y < self.y_bound[0] or spec_y > self.y_bound[1])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._prev_action = -1

        if self.random_reset:
            self.rx  = self.np_random.uniform(-0.5, 0.5)       # spec X
            self.ry  = self.np_random.uniform(-0.3, 0.3)       # spec Z (vertical)
            self.rz  = self.np_random.uniform(-self.start_distance - 0.5,
                                               -self.start_distance + 0.5)  # spec -Y
            self.yaw = self.np_random.uniform(-40.0, 40.0)
        else:
            self.rx  = 0.0
            self.ry  = 0.0
            self.rz  = -self.start_distance
            self.yaw = 0.0

        obs = self._get_observation()
        # initialise prev_error for potential-based reward
        self._prev_error = self._compute_total_error(obs)
        return obs, {}

    def step(self, action):
        theta = np.radians(self.yaw)
        fwd   = np.array([np.sin(theta), 0.0, np.cos(theta)])
        right = np.array([np.cos(theta), 0.0, -np.sin(theta)])

        # Action mapping (spec coordinates):
        # 0: +X (slide right),  1: -X (slide left)
        # 2: +Y (approach wall), 3: -Y (recede from wall)
        # 4: +Z (rise),          5: -Z (lower)
        # 6: Rotate CW,          7: Rotate CCW

        if   action == 0:  # Move +X (Right)
            self.rx += right[0] * self.move_step
            self.rz += right[2] * self.move_step
        elif action == 1:  # Move -X (Left)
            self.rx -= right[0] * self.move_step
            self.rz -= right[2] * self.move_step
        elif action == 2:  # Move +Y (Approach wall / forward)
            self.rx += fwd[0] * self.move_step
            self.rz += fwd[2] * self.move_step
        elif action == 3:  # Move -Y (Recede from wall / backward)
            self.rx -= fwd[0] * self.move_step
            self.rz -= fwd[2] * self.move_step
        elif action == 4:  # Move +Z (Rise / Up)
            self.ry += self.move_step
        elif action == 5:  # Move -Z (Lower / Down)
            self.ry -= self.move_step
        elif action == 6:  # Rotate CW
            self.yaw += self.rotate_step
        elif action == 7:  # Rotate CCW
            self.yaw -= self.rotate_step

        # ── clamp vertical position ─────────────────────────
        self.ry = np.clip(self.ry, -0.60, 0.60)

        self.current_step += 1
        obs    = self._get_observation()
        reward = self._compute_reward(obs, action)

        self._prev_action = action

        terminated = self._aligned
        truncated  = self.current_step >= self.max_steps

        return obs, reward, terminated, truncated, {}

    # ==========================================================
    #  Rendering & observation
    # ==========================================================

    def _render_frame(self):
        """Render the camera view and detect ArUco markers."""
        frame = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        if self._wall_visible():
            H = self._wall_homography()
            cv2.warpPerspective(
                self.wall_texture, H, (self.W, self.H),
                dst=frame, borderMode=cv2.BORDER_TRANSPARENT,
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        return frame, corners, ids

    def _get_observation(self):
        """Build 24-dim observation: 16 pixel offsets + 8 marker areas."""
        frame, corners, ids = self._render_frame()

        marker_centers_px = np.zeros((8, 2), dtype=np.float32)
        marker_areas_px   = np.zeros(8, dtype=np.float32)
        marker_visible    = np.zeros(8, dtype=np.int8)

        if corners is not None and ids is not None:
            for i, c in enumerate(corners):
                idx = int(ids[i][0])
                if 0 <= idx < 8:
                    pts = c[0]
                    marker_centers_px[idx] = [
                        pts[:, 0].mean(), pts[:, 1].mean()
                    ]
                    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
                    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
                    marker_areas_px[idx] = (x2 - x1) * (y2 - y1)
                    marker_visible[idx] = 1

        # ── pixel offsets (normalised) ───────────────────────
        # offset = (marker_center - ref_center) / image_dim
        pixel_offsets = np.zeros((8, 2), dtype=np.float32)
        pixel_offsets[:, 0] = (marker_centers_px[:, 0] - self.ref_centers_px[:, 0]) / self.W
        pixel_offsets[:, 1] = (marker_centers_px[:, 1] - self.ref_centers_px[:, 1]) / self.H
        # zero out invisible markers
        pixel_offsets[marker_visible == 0] = 0.0

        # ── marker areas (normalised) ────────────────────────
        norm_areas = marker_areas_px / self.max_marker_area
        norm_areas[marker_visible == 0] = 0.0

        return {
            "pixel_offsets":   pixel_offsets,       # (8, 2) normalised
            "marker_areas":    norm_areas,           # (8,) normalised
            "marker_visible":  marker_visible,       # (8,) binary
            # raw values kept for reward computation & visualisation
            "_centers_px":     marker_centers_px,
            "_areas_px":       marker_areas_px,
        }

    # ==========================================================
    #  Error computation
    # ==========================================================

    def _compute_total_error(self, obs):
        """Total Euclidean error E = Σ sqrt((xi-xt)² + (yi-yt)²) in pixels."""
        vis = obs["marker_visible"]
        centers_px = obs["_centers_px"]
        E = 0.0
        for i in range(8):
            if vis[i]:
                dx = centers_px[i, 0] - self.ref_centers_px[i, 0]
                dy = centers_px[i, 1] - self.ref_centers_px[i, 1]
                E += np.sqrt(dx**2 + dy**2)
            else:
                # Invisible marker: add a large penalty error
                E += np.sqrt(self.W**2 + self.H**2)
        return float(E)

    # ==========================================================
    #  Reward
    # ==========================================================

    def _compute_reward(self, obs, action):
        """Reward function as per DQN 8-Marker Alignment spec.

        Potential-Based: 10.0 × (E_{t-1} − E_t)
        Terminal Success: +500 if all markers within threshold
        Safety Penalty:  -100 if any marker lost or robot out of XY bounds
        Anti-Jiggle:     -5.0 if action reverses the previous action
        Step Cost:       -0.5 per action
        """
        vis = obs["marker_visible"]
        num_visible = int(vis.sum())
        centers_px = obs["_centers_px"]

        # ── total Euclidean error ────────────────────────────
        E = self._compute_total_error(obs)
        self._total_error = E

        # ── per-marker pixel distances (for alignment check) ─
        pixel_dists = np.zeros(8, dtype=np.float32)
        for i in range(8):
            if vis[i]:
                dx = centers_px[i, 0] - self.ref_centers_px[i, 0]
                dy = centers_px[i, 1] - self.ref_centers_px[i, 1]
                pixel_dists[i] = np.sqrt(dx**2 + dy**2)
            else:
                pixel_dists[i] = 999.0
        self._pixel_dists = pixel_dists
        self._max_pixel_dist = float(pixel_dists.max())

        # ── alignment check ──────────────────────────────────
        self._aligned = (
            num_visible == 8
            and float(pixel_dists.max()) <= self.align_pixel_thresh
        )

        # ── potential-based reward ───────────────────────────
        potential_reward = 10.0 * (self._prev_error - E)
        self._prev_error = E

        # ── total reward accumulation ────────────────────────
        reward = potential_reward

        # ── terminal success bonus ───────────────────────────
        if self._aligned:
            reward += 500.0

        # ── safety / visibility penalty ──────────────────────
        if num_visible < 8 or self._out_of_bounds():
            reward -= 100.0

        # ── anti-jiggle penalty ──────────────────────────────
        # Penalise immediately reversing the previous action
        if action != -1 and self._prev_action != -1:
            rev_map = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6}
            if action == rev_map.get(self._prev_action):
                reward -= 5.0

        # ── step cost ────────────────────────────────────────
        reward -= 0.5

        return reward

    # ==========================================================
    #  Flatten observation for RL training
    # ==========================================================

    @staticmethod
    def flatten_obs(obs):
        """Convert dict observation → flat 24-float numpy array.

        Layout: [offset_x0, offset_y0, ... offset_x7, offset_y7,
                 area_0, area_1, ... area_7]

        16 pixel offsets + 8 marker areas = 24 dimensions.
        """
        offsets = obs["pixel_offsets"]      # (8, 2)
        areas   = obs["marker_areas"]       # (8,)
        return np.concatenate([offsets.flatten(), areas])

    # ==========================================================
    #  Visualisation
    # ==========================================================

    def visualize(self, reward=None):
        frame, corners, ids = self._render_frame()

        # ── detected markers (green) ─────────────────────────
        if corners is not None:
            for i, c in enumerate(corners):
                pts = c[0].astype(int)
                cx  = int(pts[:, 0].mean())
                cy  = int(pts[:, 1].mean())
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                if ids is not None:
                    mid = int(ids[i][0])
                    cv2.putText(frame, f"A:{mid}",
                                (cx - 20, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

        # ── reference boxes (red) ────────────────────────────
        for i in range(8):
            bx1, by1, bx2, by2 = self.ref_boxes_px[i].astype(int)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
            rcx, rcy = self.ref_centers_px[i].astype(int)
            cv2.putText(frame, f"R:{i}",
                        (rcx - 15, rcy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)

        # ── HUD ──────────────────────────────────────────────
        if reward is not None:
            cv2.putText(frame, f"Reward: {reward:.3f}",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 255), 2)

        if hasattr(self, '_current_action_name') and self._current_action_name:
            cv2.putText(frame, f"Action: {self._current_action_name}",
                        (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)

        # Show spec coordinates: X=lateral, Y=depth, Z=vertical
        spec_x = self.rx
        spec_y = self._spec_y()
        spec_z = self.ry
        cv2.putText(
            frame,
            f"X={spec_x:.2f}  Y={spec_y:.2f}  "
            f"Z={spec_z:.2f}  Yaw={self.yaw:.1f}",
            (30, self.H - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
        )

        # Show total error
        if hasattr(self, '_total_error'):
            cv2.putText(
                frame,
                f"Error: {self._total_error:.1f}px",
                (30, self.H - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1,
            )

        cv2.imshow("Robot Wall Environment", frame)
        cv2.waitKey(1)

    def print_observation(self, obs, reward):
        """Helper to print current observation state."""
        vis = obs["marker_visible"]
        num_vis = int(vis.sum())
        print(f"  Visible Markers : {num_vis}/8")
        if hasattr(self, '_max_pixel_dist'):
            print(f"  Max Pixel Dist  : {self._max_pixel_dist:.2f} px")
        if hasattr(self, '_total_error'):
            print(f"  Total Error     : {self._total_error:.1f} px")
        print(f"  Aligned         : {self._aligned}")

        # Print marker areas
        areas = obs["marker_areas"]
        area_str = "  ".join(f"{a:.5f}" if vis[i] else " --  " for i, a in enumerate(areas))
        print(f"  Marker Areas    : {area_str}")

        if reward is not None:
            print(f"  Reward          : {reward:.4f}")


# ================= MANUAL CONTROL ==============================

if __name__ == "__main__":

    env = RobotWallEnv()
    obs, _ = env.reset()

    print("\n=== Robot Wall Environment ===")
    print("  Coordinate System: X=Lateral, Y=Depth, Z=Vertical")
    print("  D / A   — Move +X / -X (Right / Left)")
    print("  W / S   — Move +Y / -Y (Approach / Recede)")
    print("  R / F   — Move +Z / -Z (Rise / Lower)")
    print("  E / Q   — Rotate CW / CCW (Yaw around Z)")
    print("  ESC     — Quit\n")

    action_names = [
        "Move +X (Right)", "Move -X (Left)",
        "Move +Y (Approach)", "Move -Y (Recede)",
        "Move +Z (Rise)", "Move -Z (Lower)",
        "Rotate CW", "Rotate CCW",
    ]

    # print initial state
    reward = env._compute_reward(obs, action=-1)
    env.print_observation(obs, reward)

    while True:
        env.visualize(reward)

        key = cv2.waitKeyEx(0)

        if key == 27:
            break
        elif key == ord('d'):
            action = 0   # +X
        elif key == ord('a'):
            action = 1   # -X
        elif key == ord('w'):
            action = 2   # +Y
        elif key == ord('s'):
            action = 3   # -Y
        elif key == ord('r'):
            action = 4   # +Z
        elif key == ord('f'):
            action = 5   # -Z
        elif key == ord('e'):
            action = 6   # CW
        elif key == ord('q'):
            action = 7   # CCW
        else:
            continue

        obs, reward, _, _, _ = env.step(action)
        print(f"\n>>> Action {action}: {action_names[action]}")
        env.print_observation(obs, reward)

    cv2.destroyAllWindows()
