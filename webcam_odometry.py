import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class CameraPoses:
    def __init__(self, intrinsic, dist_coeffs):
        self.K = intrinsic
        self.dist_coeffs = dist_coeffs
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=50, key_size=15, multi_probe_level=1)
        search_params = dict(checks=100)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    def _form_transf(self, R, t):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return T

    def get_matches(self, img1, img2):
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)
        if des1 is None or des2 is None:
            return None, None
        matches = self.flann.knnMatch(des1, des2, k=2)

        good = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.5 * n.distance:
                good.append(m)

        if len(good) < 8:
            return None, None

        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2

    def get_pose(self, q1, q2):
        E, mask = cv2.findEssentialMat(q1, q2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or mask is None:
            return None, 0
        inlier_ratio = float(np.sum(mask)) / len(mask)
        if inlier_ratio < 0.4:  # reject low-quality estimations
            return None, inlier_ratio
        _, R, t, _ = cv2.recoverPose(E, q1, q2, self.K)
        T = self._form_transf(R, t)
        return T, inlier_ratio


# === Load calibration ===
with np.load("camera_calibration_webcam.npz") as X:
    K = X["K"]
    dist = X["dist"]

vo = CameraPoses(K, dist)
cur_pose = np.eye(4)
trajectory = []
camera_poses = []

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not accessible")
    exit()

# Smoothing
alpha = 0.05
prev_x, prev_y, prev_z = 0.0, 0.0, 0.0
old_gray = None
frame_id = 0

# Thresholds
motion_thresh = 1.5
motion_min = 0.005

while True:
    ret, frame = cap.read()
    if not ret:
        break

    undistorted = cv2.undistort(frame, K, dist)
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

    if old_gray is not None:
        q1, q2 = vo.get_matches(old_gray, gray)
        if q1 is not None and q2 is not None:
            T, inlier_ratio = vo.get_pose(q1, q2)
            if T is not None:
                motion = T[:3, 3]
                motion_norm = np.linalg.norm(motion)

                if motion_min < motion_norm < motion_thresh:
                    cur_pose = cur_pose @ T

                    # Smooth X, Y, Z
                    x = alpha * cur_pose[0, 3] + (1 - alpha) * prev_x
                    y = alpha * cur_pose[1, 3] + (1 - alpha) * prev_y
                    z = alpha * cur_pose[2, 3] + (1 - alpha) * prev_z
                    prev_x, prev_y, prev_z = x, y, z

                    trajectory.append([x, y, z])
                    camera_poses.append(cur_pose.copy())

                else:
                    print(f"[INFO] Frame {frame_id} skipped: motion_norm={motion_norm:.2f}, inlier_ratio={inlier_ratio:.2f}")
            else:
                print(f"[INFO] Pose estimation failed at frame {frame_id}")
    else:
        print(f"[INFO] First frame initialized.")

    old_gray = gray.copy()
    frame_id += 1

    # Display
    disp = undistorted.copy()
    cv2.putText(disp, f"Frame: {frame_id}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(disp, f"X: {cur_pose[0, 3]:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(disp, f"Y: {cur_pose[1, 3]:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(disp, f"Z: {cur_pose[2, 3]:.2f}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Visual Odometry", disp)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# === Plot Trajectory ===
trajectory = np.array(trajectory)
if len(trajectory) > 0:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label="Trajectory", color="blue")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("Camera Trajectory")
    plt.tight_layout()
    plt.show()
else:
    print("No trajectory recorded.")