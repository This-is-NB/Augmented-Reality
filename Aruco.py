from inspect import Parameter
from argon2 import Parameters
import cv2
from pip import main
from pytest import mark
from torch import imag
import cv2.aruco as aruco
import numpy as np
import os

def FindArucoMarkers(img, markersize = 7,totalmarkers = 250, draw=True):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_{markersize}X{markersize}_{totalmarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(gray,arucoDict, parameters=arucoParam)
    # print(ids)
    if draw:
        aruco.drawDetectedMarkers(img,bboxs)
    return [bboxs,ids]

def ArucoAugment(bbox, ids, img, imgaug, drawID=True):
    
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]
    h,w, c =  imgaug.shape
    pts1 = np.array([tl,tr,br,bl])
    pts2 = np.float32([[0,0],[w,0],[w,h],[0,h]])
    matrix , _= cv2.findHomography(pts2,pts1)
    outputimage = cv2.warpPerspective(imgaug,matrix,(img.shape[1],img.shape[0]))
    cv2.fillConvexPoly(img, pts1.astype(int),0)
    outputimage += img
    
    if drawID:
        cv2.putText(outputimage,str(ids), np.array(tl).astype(int) , cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,0,255), 2)
       
    return outputimage
    
def main():
    cap = cv2.VideoCapture(0)
    Video = cv2.VideoCapture("Video.mkv")
    while True:
        ret, frame = cap.read()
        ret, Videoframe = Video.read()
        # aug = cv2.imread("AceOfSpades.jpg")
        arucoFound = FindArucoMarkers(frame)
        if len(arucoFound[0]) != 0:
            for bbox , ids in zip(arucoFound[0], arucoFound[1]):
                frame =ArucoAugment(bbox,ids,frame,Videoframe)
        cv2.imshow("frame",frame)
        key = cv2.waitKey(1)
        if key == 27: break
    cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()