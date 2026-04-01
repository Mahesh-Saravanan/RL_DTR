import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PerspectiveAlignmentEnv(gym.Env):

    def __init__(self):

        self.W = 1024
        self.H = 768
        self.ratio = 1.687

        self.cols = 4
        self.rows = 2
        self.marker_ratio = 0.2

        self.move_step = 15
        self.scale_step = 0.05
        self.tilt_step = 5

        self.min_scale = 0.2
        self.max_scale = 3.0

        self.max_distance = np.sqrt(self.W**2 + self.H**2)

        canvas_area = self.W * self.H
        target_area = 0.60 * canvas_area

        self.box_h = int(np.sqrt(target_area / self.ratio))
        self.box_w = int(self.ratio * self.box_h)

        self.cx = self.W // 2
        self.cy = self.H // 2

        self.ref_x1 = self.cx - self.box_w // 2
        self.ref_y1 = self.cy - self.box_h // 2

        self.cell_w = self.box_w // self.cols
        self.cell_h = self.box_h // self.rows
        self.marker_size = int(min(self.cell_w, self.cell_h) * self.marker_ratio)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict,
            cv2.aruco.DetectorParameters()
        )

        self.action_space = spaces.Discrete(8)

        self.observation_space = spaces.Dict({
            "marker_centers": spaces.Box(-1e4, 1e4, (8, 2), dtype=np.float32),
            "marker_boxes": spaces.Box(-1e4, 1e4, (8, 4), dtype=np.float32),
            "ref_centers": spaces.Box(-1e4, 1e4, (8, 2), dtype=np.float32),
            "ref_boxes": spaces.Box(-1e4, 1e4, (8, 4), dtype=np.float32),
        })

        self.reset()

    # =====================================================

    def reset(self, seed=None, options=None):

        self.tx = 0
        self.ty = 0
        self.scale_factor = 1.0
        self.tilt_x = 0

        obs = self._get_observation()
        return obs, {}

    # =====================================================

    def step(self, action):

        if action == 0:
            self.tx -= self.move_step
        elif action == 1:
            self.tx += self.move_step
        elif action == 2:
            self.ty -= self.move_step
        elif action == 3:
            self.ty += self.move_step
        elif action == 4:
            self.scale_factor = max(self.min_scale,
                                    self.scale_factor - self.scale_step)
        elif action == 5:
            self.scale_factor = min(self.max_scale,
                                    self.scale_factor + self.scale_step)
        elif action == 6:
            self.tilt_x -= self.tilt_step
        elif action == 7:
            self.tilt_x += self.tilt_step

        obs = self._get_observation()
        reward = self._compute_reward(obs)

        return obs, reward, False, False, {}

    # =====================================================

    def _render_frame(self):

        base = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        scaled_w = int(self.box_w * self.scale_factor)
        scaled_h = int(self.box_h * self.scale_factor)

        disp_cx = self.cx + self.tx
        disp_cy = self.cy + self.ty

        x1 = disp_cx - scaled_w // 2
        y1 = disp_cy - scaled_h // 2
        x2 = disp_cx + scaled_w // 2
        y2 = disp_cy + scaled_h // 2

        cv2.rectangle(base, (x1, y1), (x2, y2), (255, 255, 255), -1)

        scaled_cell_w = scaled_w // self.cols
        scaled_cell_h = scaled_h // self.rows
        scaled_marker_size = int(
            min(scaled_cell_w, scaled_cell_h) * self.marker_ratio
        )

        marker_id = 0
        for r in range(self.rows):
            for c in range(self.cols):

                cell_cx = x1 + c * scaled_cell_w + scaled_cell_w // 2
                cell_cy = y1 + r * scaled_cell_h + scaled_cell_h // 2

                marker = cv2.aruco.generateImageMarker(
                    self.aruco_dict,
                    marker_id,
                    scaled_marker_size
                )
                marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

                mx1 = cell_cx - scaled_marker_size // 2
                my1 = cell_cy - scaled_marker_size // 2
                mx2 = mx1 + scaled_marker_size
                my2 = my1 + scaled_marker_size

                if 0 <= mx1 < self.W and 0 <= my1 < self.H and mx2 < self.W and my2 < self.H:
                    base[my1:my2, mx1:mx2] = marker

                marker_id += 1

        src = np.float32([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
        dst = np.float32([
            [x1+self.tilt_x,y1],
            [x2-self.tilt_x,y1],
            [x2+self.tilt_x,y2],
            [x1-self.tilt_x,y2]
        ])

        M = cv2.getPerspectiveTransform(src,dst)
        warped = cv2.warpPerspective(base,M,(self.W,self.H))

        gray = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
        corners,ids,_ = self.detector.detectMarkers(gray)

        return warped, corners, ids

    # =====================================================

    def _get_observation(self):

        frame, corners, ids = self._render_frame()

        marker_centers = np.zeros((8,2),dtype=np.float32)
        marker_boxes = np.zeros((8,4),dtype=np.float32)
        ref_centers = np.zeros((8,2),dtype=np.float32)
        ref_boxes = np.zeros((8,4),dtype=np.float32)

        if corners is not None:
            for i,c in enumerate(corners):
                pts = c[0]
                cx = pts[:,0].mean()
                cy = pts[:,1].mean()
                x1,y1 = pts.min(axis=0)
                x2,y2 = pts.max(axis=0)

                marker_centers[i] = [cx,cy]
                marker_boxes[i] = [x1,y1,x2,y2]

        idx=0
        for r in range(self.rows):
            for c in range(self.cols):

                cell_cx = self.ref_x1 + c*self.cell_w + self.cell_w//2
                cell_cy = self.ref_y1 + r*self.cell_h + self.cell_h//2

                mx1 = cell_cx - self.marker_size//2
                my1 = cell_cy - self.marker_size//2
                mx2 = mx1 + self.marker_size
                my2 = my1 + self.marker_size

                ref_centers[idx] = [cell_cx,cell_cy]
                ref_boxes[idx] = [mx1,my1,mx2,my2]
                idx+=1

        return {
            "marker_centers": marker_centers,
            "marker_boxes": marker_boxes,
            "ref_centers": ref_centers,
            "ref_boxes": ref_boxes,
        }

    # =====================================================

    def _compute_reward(self,obs):

        mc = obs["marker_centers"]
        rc = obs["ref_centers"]

        dist = np.linalg.norm(mc-rc,axis=1)
        D = 1 - np.mean(dist) / self.max_distance
        D = np.clip(D,0,1)

        mb = obs["marker_boxes"]
        rb = obs["ref_boxes"]

        ious=[]
        for i in range(8):
            xa=max(mb[i][0],rb[i][0])
            ya=max(mb[i][1],rb[i][1])
            xb=min(mb[i][2],rb[i][2])
            yb=min(mb[i][3],rb[i][3])

            inter=max(0,xb-xa)*max(0,yb-ya)
            area1=max(1,(mb[i][2]-mb[i][0])*(mb[i][3]-mb[i][1]))
            area2=max(1,(rb[i][2]-rb[i][0])*(rb[i][3]-rb[i][1]))
            union=area1+area2-inter
            iou=inter/union if union>0 else 0
            ious.append(iou)

        I = np.mean(ious)

        return D + I - 1

    # =====================================================

    def visualize(self,reward=None):

        frame,corners,ids = self._render_frame()

        # Draw detected markers + ID
        if corners is not None:
            for i,c in enumerate(corners):
                pts = c[0].astype(int)
                cx = int(pts[:,0].mean())
                cy = int(pts[:,1].mean())

                cv2.polylines(frame,[pts],True,(0,255,0),2)

                marker_id = int(ids[i][0])
                cv2.putText(frame,
                            f"Aruco:{marker_id}",
                            (cx-30,cy-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,(0,255,0),2)

        # Draw reference boxes + ID
        idx=0
        for r in range(self.rows):
            for c in range(self.cols):
                cell_cx = self.ref_x1 + c*self.cell_w + self.cell_w//2
                cell_cy = self.ref_y1 + r*self.cell_h + self.cell_h//2
                mx1 = cell_cx - self.marker_size//2
                my1 = cell_cy - self.marker_size//2
                mx2 = mx1 + self.marker_size
                my2 = my1 + self.marker_size

                cv2.rectangle(frame,(mx1,my1),(mx2,my2),(0,0,255),2)

                cv2.putText(frame,
                            f"Ref:{idx}",
                            (cell_cx-30,cell_cy+20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,(0,0,255),2)

                idx+=1

        if reward is not None:
            cv2.putText(frame,f"Reward: {reward:.3f}",
                        (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,(0,255,255),2)

        cv2.imshow("RL Environment",frame)
        cv2.waitKey(1)


# ================= MANUAL CONTROL =================

if __name__ == "__main__":

    env = PerspectiveAlignmentEnv()
    obs,_ = env.reset()

    while True:

        reward = env._compute_reward(obs)
        env.visualize(reward)

        key = cv2.waitKeyEx(0)

        if key == 27:
            break
        elif key == 2424832:
            action=0
        elif key == 2555904:
            action=1
        elif key == 2490368:
            action=2
        elif key == 2621440:
            action=3
        elif key == 2228224:
            action=4
        elif key == 2162688:
            action=5
        elif key == ord('a'):
            action=6
        elif key == ord('d'):
            action=7
        else:
            continue

        obs,reward,_,_,_ = env.step(action)

    cv2.destroyAllWindows()
