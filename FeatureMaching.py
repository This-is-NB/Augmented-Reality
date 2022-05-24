from heapq import heapify
import cv2
import numpy as np

MIN_MATCHES = 20
orb = cv2.ORB_create(nfeatures = 5000)

index_params = dict(algorithm = 1, trees = 3)
search_param = dict(checks = 100)
flann = cv2.FlannBasedMatcher(index_params,search_param)

def load_input():
    img = cv2.imread("./main_image/img_taj.jpg")
    print(np.array(img.shape))
    img = cv2.resize(img,np.array(img.shape[:2][::-1])//8, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    keypoints, descriptors = orb.detectAndCompute(gray,None)
    return gray , keypoints , descriptors


def compute_matches(descriptor_input, descriptor_output):
    if len(descriptor_input) != 0 and len(descriptor_output) != 0:
        matches = flann.knnMatch(np.asarray(descriptor_input,np.float32) ,np.asarray(descriptor_output,np.float32),k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good.append([m])
        return good
    else: return None
    
    
if __name__ == "__main__":
    img, img_keypoints, img_descriptors = load_input()
    cap = cv2.VideoCapture(0)
    ret,frame = cap.read()
    while(ret):
        ret,frame = cap.read()
         
        if len(img_keypoints) < MIN_MATCHES:
            continue
        frame = cv2.resize(frame , (700,600))
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        output_keypoints,output_descriptors = orb.detectAndCompute(grayframe,None)
        matches = compute_matches(img_descriptors,output_descriptors)
        
        if matches != None:
            output_final = cv2.drawMatchesKnn(img,img_keypoints,frame,output_keypoints,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow("Final" , output_final)
        else:
            cv2.imshow("Final",frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
cap.release()
cv2.destroyAllWindows()
        
            
