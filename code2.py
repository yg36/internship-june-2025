# corners detection btw two images using bf matcher and norm_hamming basic code 2
import cv2
import numpy as np
import os

# Load a sequence of images (simulate navigation)
# You can replace these with your own image paths
base_path = r"E:\YG\codes\Python\IITM\vision based navigation"
image_paths = [os.path.join(base_path, "frame1.jpg"), os.path.join(base_path, "frame2.jpg")]

# Load the first and second frames
img1 = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(image_paths[1], cv2.IMREAD_GRAYSCALE)

# exception handling if images are not found
if img1 is None or img2 is None:
    print("Error loading images. Check file names and paths.")
else:
    print("Images loaded successfully!")

# Initialize ORB detector (can also use SIFT or AKAZE)
orb = cv2.ORB_create()

# Step 1: Detect keypoints and descriptors
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Step 2: Match features using Brute Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Step 3: Sort and select good matches
matches = sorted(matches, key=lambda x: x.distance)
good_matches = matches[:50]

# Step 4: Extract point coordinates from good matches
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# Step 5: Estimate Essential Matrix (assuming known camera intrinsics)
focal_length = 718.8560  # example value, change to match your camera
center = (607.1928, 185.2157)  # example value, change to match your camera
K = np.array([[focal_length, 0, center[0]],
              [0, focal_length, center[1]],
              [0, 0, 1]])

E, mask = cv2.findEssentialMat(pts2, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# Step 6: Recover relative pose (R, t) from Essential Matrix
_, R, t, mask = cv2.recoverPose(E, pts2, pts1, K)

print("Estimated Rotation Matrix:")
print(R)

print("\nEstimated Translation Vector:")
print(t)

# Optional: Visualize Matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None)
cv2.imshow("Feature Matches", img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
