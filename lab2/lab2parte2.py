import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time 


MIN_MATCH_COUNT = 10
FPS=10

cap1 = cv.VideoCapture(0)
cap2 = cv.VideoCapture(1)
sift = cv.SIFT_create()

while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if cv.waitKey(1) == ord('q'):
        break

    if not ret1 or not ret2:
        print("Erro ao capturar os quadros.")
        break

    kp1, des1 = sift.detectAndCompute(frame1,None)
    kp2, des2 = sift.detectAndCompute(frame2,None)

    
    # Exibe os feeds em janelas separadas
    #cv.imshow('Camera 1', frame1)
    #cv.imshow('Camera 2', frame2)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
    
        h,w = frame1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
    
        frame2 = cv.polylines(frame2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
    
    img3 = cv.drawMatches(frame1,kp1,frame2,kp2,good,None,**draw_params)
    
    plt.imshow(img3, 'gray'),plt.show()
    time.sleep(1/FPS)
# Initiate SIFT detector


# find the keypoints and descriptors with SIFT

