# this code is used for validation of calibration. input: image and camera matrix and square sizes. output: distance to checkerboard.
import cv2
import numpy as np

# === Load calibration data ===
data = np.load('camera_calibration_webcam.npz')
K = data['camera_matrix']
dist = data['dist_coeffs']

# === Checkerboard settings ===
pattern_size = (8, 5)  # inner corners (columns, rows)
square_size_mm = 32    # real square size in mm
test_img_path = r"calib_images\frame_019.jpg"  # <-- Change to your test image path

# === Prepare object points in real-world units (mm) ===
objp = np.zeros((pattern_size[1]*pattern_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size_mm

# === Load and undistort test image ===
img = cv2.imread(test_img_path)
if img is None:
    print(f"Could not read image: {test_img_path}")
    exit(1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
undistorted = cv2.undistort(img, K, dist)
gray_undist = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

# === Find corners ===
ret, corners = cv2.findChessboardCorners(gray_undist, pattern_size, None)
if not ret:
    print("Checkerboard not detected in test image.")
    exit(1)

# Refine corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corners_refined = cv2.cornerSubPix(gray_undist, corners, (11, 11), (-1, -1), criteria)

# === Estimate pose ===
retval, rvec, tvec = cv2.solvePnP(objp, corners_refined, K, dist)
if not retval:
    print("solvePnP failed.")
    exit(1)

# === tvec is the translation vector from camera to checkerboard origin ===
distance_to_checkerboard_mm = np.linalg.norm(tvec)
print(f"Estimated distance from camera to checkerboard origin: {distance_to_checkerboard_mm:.2f} mm")

# === Optional: Visualize ===
cv2.drawChessboardCorners(undistorted, pattern_size, corners_refined, ret)
cv2.imshow('Undistorted Checkerboard', undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()