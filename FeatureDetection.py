from heapq import heapify
import cv2
import numpy as np
img = cv2.imread("./main_image/img_taj.jpg")
print(np.array(img.shape))
img = cv2.resize(img,np.array(img.shape[:2][::-1])//6, interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures = 1000)
keypoints, descriptors = orb.detectAndCompute(gray,None)

final_keypoints = cv2.drawKeypoints(gray,keypoints,img,(0,200,0))

cv2.imshow("keypoints",final_keypoints)
cv2.waitKey()
