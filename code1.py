# frame transformations basic code 1
import numpy as np
import cv2

img = cv2.imread("unerwater.jpg",cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(img, None)
theta = np.deg2rad(30)
R = np.array([[np.cos(theta), -np.sin(theta), 0],
              [np.sin(theta), np.cos(theta), 0],
              [0,0,1]])

T = np.array([[2], [-1], [8],])
T_robot = np.vstack((np.hstack((R,T)), [0,0,0,1]))

P = np.array([[1],[4],[0], [1]])
P_world = T_robot @ P
print("transformed Point:" , P_world)