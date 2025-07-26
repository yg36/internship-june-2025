# this code captures two frame using webcam and then draw matches(if they exist). basic code 3

import cv2
import numpy as np

# === STEP 1: Capture 2 frames from webcam ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not detected.")
    exit()

print("Press SPACE to capture the first frame.")
while True:
    ret, frame1 = cap.read()
    cv2.imshow("Capture Frame 1", frame1)
    if cv2.waitKey(1) == 32:  # SPACE
        break

print("Now move the camera slightly and press SPACE again.")
while True:
    ret, frame2 = cap.read()
    cv2.imshow("Capture Frame 2", frame2)
    if cv2.waitKey(1) == 32:
        break

cap.release()
cv2.destroyAllWindows()

# Convert to grayscale
img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# === STEP 2: ORB feature detection ===
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# === STEP 3: Match features using BFMatcher ===
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
good_matches = matches[:50]

# === STEP 4: Extract matched points ===
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# === STEP 5: Dummy camera intrinsics ===
focal_length = 700.0
center = (img1.shape[1] / 2, img1.shape[0] / 2)
K = np.array([[focal_length, 0, center[0]],
              [0, focal_length, center[1]],
              [0, 0, 1]])

# === STEP 6: Estimate Essential Matrix and Pose ===
E, mask = cv2.findEssentialMat(pts2, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
_, R, t, mask = cv2.recoverPose(E, pts2, pts1, K)

print("\n📐 Estimated Rotation Matrix (R):\n", R)
print("\n➡️ Estimated Translation Vector (t):\n", t)

# === STEP 7: Show Movement and Direction ===
total_movement = 0
print("\n📍 Feature Movement Details:")
for i, (p1, p2) in enumerate(zip(pts1, pts2)):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dist = np.sqrt(dx**2 + dy**2)
    total_movement += dist

    dir_x = "right" if dx > 0 else "left" if dx < 0 else "no change"
    dir_y = "down" if dy > 0 else "up" if dy < 0 else "no change"

    print(f"  🔹 Point {i+1}: moved {dist:.2f} px → Direction: {abs(dx):.2f}px {dir_x}, {abs(dy):.2f}px {dir_y}")

avg_movement = total_movement / len(pts1) if len(pts1) > 0 else 0
print(f"\n📏 Average pixel movement: {avg_movement:.2f} px")

# === STEP 8: Optional — Draw matches ===
match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None)
cv2.imshow("Matches", match_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
