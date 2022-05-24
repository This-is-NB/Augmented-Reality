from heapq import heapify
import cv2
from cv2 import FlannBasedMatcher
import numpy as np
from sklearn.metrics import auc

MIN_MATCHES = 20
orb = cv2.ORB_create(nfeatures = 5000)

FLANN_INDEX_KDTREE = 1

index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_param = dict(checks = 100)
flann = cv2.FlannBasedMatcher(index_params,search_param)

def load_input():
    img = cv2.imread("./main_image/img_taj.jpg")
    # augment_img = cv2.imread(("AceOfSpades.jpg"))
    
    # print(np.array(img.shape))
    shape = np.array(img.shape[:2][::-1])//10
    # shape = (300,390)
    # shape = (240,360)
    print(shape)
    img = cv2.resize(img,shape, interpolation=cv2.INTER_AREA)
    # img = cv2.resize(img,(300,400), interpolation=cv2.INTER_AREA)
    # augment_img = cv2.resize(augment_img, shape)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    keypoints, descriptors = orb.detectAndCompute(gray,None)
    return gray , keypoints , descriptors ,shape
    # return gray ,augment_img, keypoints , descriptors ,shape


def compute_matches(descriptor_input, descriptor_output):
    if len(descriptor_input) != 0 and len(descriptor_output) != 0:
        matches = flann.knnMatch(np.asarray(descriptor_input,np.float32) ,np.asarray(descriptor_output,np.float32),k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.8*n.distance:
                good.append(m)
        return good
    else: return None
    
    
if __name__ == "__main__":
    img, img_keypoints, img_descriptors, img_shape = load_input()
    cap = cv2.VideoCapture(0)
    Video = cv2.VideoCapture("Video.mkv")
    ret,frame = cap.read()
    while(ret):
        ret,frame = cap.read()
        succ, videoFrame = Video.read()
        if len(img_keypoints) < MIN_MATCHES:
            continue
        frame = cv2.resize(frame , (640,480))  
        videoFrame = cv2.resize(videoFrame , img_shape)  
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        output_keypoints,output_descriptors = orb.detectAndCompute(grayframe,None)
        matches = compute_matches(img_descriptors,output_descriptors)
        
        if matches != None:
            if len(matches) > 50:
                src_pts = np.float32([img_keypoints[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
                dst_pts = np.float32([output_keypoints[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        
                M,mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,10.0)
                w,h = img_shape
                pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)        
                dst = cv2.perspectiveTransform(pts,M)
                M_aug = cv2.warpPerspective(videoFrame,M,(640,480))
                # M_aug = cv2.warpPerspective(aug_img,M,(640,480))
                
                frameb = cv2.fillConvexPoly(frame.copy(),dst.astype(int),0)
                Final = frameb + M_aug 
                # Zero background where we want to overlay

                # Add object to zeroed out space

                
                cv2.imshow("Final" , Final)
                cv2.imshow("Aug", M_aug)
                # cv2.imshow("frame", frame)
                # cv2.imshow("frameb", frame_b)
                
                
                                                                                  
                # frame[M_aug>0]=0                                                                              
                # frame += aug_img*(M_aug>0)  
                # cv2.imshow("temp", frame)
                 
                
                # cv2.imshow("Input", aug_img)
            else:
                cv2.imshow("Final",frame)
        else:
            cv2.imshow("Final",frame)
        # cv2.imshow("Video",videoFrame)
        key = cv2.waitKey(1)
        if key == 27:
            break
cap.release()
cv2.destroyAllWindows()
        
            
