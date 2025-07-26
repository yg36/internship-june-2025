# this code will capture images from your webcam when you press space. 
import cv2
import numpy as np
import os
import glob

# === Configuration ===
pattern_size = (8, 5)  # inner corners (columns, rows)
square_size = 32  # any unit
calib_dir = "calib_images"
os.makedirs(calib_dir, exist_ok=True)

# === Step 1: Capture and Save Images ===
cap = cv2.VideoCapture(0)
print("Press SPACE to capture images. Press ESC to finish.")

img_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    disp = frame.copy()
    cv2.putText(disp, f"Captured: {img_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Webcam - Calibration Capture", disp)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key to break
        break
    elif key == 32:  # SPACE key to save image
        filename = os.path.join(calib_dir, f"frame_{img_count:03d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        img_count += 1

cap.release()
cv2.destroyAllWindows()

# === Step 2: Load Images and Detect Corners ===
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D real world points
imgpoints = []  # 2D image points

images = sorted(glob.glob(os.path.join(calib_dir, "*.jpg")))
if len(images) == 0:
    print("No images found. Exiting.")
    exit()

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        print(f"✔ Corners found in {fname}")
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
        cv2.imshow("Corners", img)
        cv2.waitKey(300)
    else:
        print(f"✘ Corners NOT found in {fname}")

cv2.destroyAllWindows()

# === Step 3: Calibrate Camera ===
if len(objpoints) == 0:
    print("No valid checkerboard detections. Cannot calibrate.")
    exit()

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# === Step 4: Save Calibration ===
print("\n=== Calibration Result ===")
print("Camera Matrix (K):\n", K)
print("\nDistortion Coefficients:\n", dist.ravel())
np.savez("camera_calibration_webcam.npz", K=K, dist=dist)
print("\n✅ Saved calibration to 'camera_calibration_data.npz'")
