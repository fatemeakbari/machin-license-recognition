import cv2
import numpy as np
import math
from .checkCandid import check
import os

def findCandidate(img,i):

    img = cv2.resize(img, (int((400 / img.shape[0]) * img.shape[1]), 400))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('./res/'+str(i)+'img_gray.jpg', img_gray)

    sobelx_img = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    cv2.imwrite('./res/'+str(i)+'sobelx_img.jpg', sobelx_img)

    abs_img = np.absolute(sobelx_img)
    cv2.imwrite('./res/'+str(i)+'abs_img.jpg', abs_img)
    _,threshold_img =  cv2.threshold(abs_img,170,255,cv2.THRESH_BINARY)

    cv2.imwrite('./res/'+str(i)+'img_gray.jpg', threshold_img)
    kernel = np.ones((2, 2), np.uint8)
    kernel2 = np.ones((10, 15), np.uint8)
    kernel3 = np.ones((2, 5), np.uint8)
    kernel4 = np.ones((3, 35), np.uint8)

    
    
    open1_img = cv2.morphologyEx(threshold_img, cv2.MORPH_OPEN, kernel)
    cv2.imwrite('./res/'+str(i)+'open1_img.jpg', open1_img)
    close_img = cv2.morphologyEx(open1_img, cv2.MORPH_CLOSE, kernel2)
    cv2.imwrite('./res/'+str(i)+'close_img.jpg', close_img)
    erode_img = cv2.erode(close_img, kernel3, iterations=1)
    cv2.imwrite('./res/'+str(i)+'erode_img.jpg', erode_img)
    dilate_img = cv2.dilate(erode_img, kernel4, iterations=1)
    cv2.imwrite('./res/'+str(i)+'dilate_img.jpg', dilate_img)
    
    (contours, hierarchy)= cv2.findContours(dilate_img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = min(8, len(contours))
   
    contours =sorted(contours, key=cv2.contourArea, reverse=True)[:length]
    # find the biggest area
    lower_blue = np.array([100,50,50])
    upper_blue = np.array([140,255,255])
    find = False
    max = 0
    candidate = []
    m=0;
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if(w > 4*h and w < 10*h):
            crop_img = img_gray[y :y + h , x  :x + w]
            if max < h*w:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

                max = h*w
                candidate.append(cv2.resize(crop_img, (200, 60)))
                find = True
    
   
    #check(candidates)
    #cv2.imwrite('./res/'+str(i)+'/erode_img.jpg',erode_img)
    if candidate:
        cv2.imwrite('./res/'+str(i)+'candid.jpg',candidate[0])
    #cv2.imwrite('./res/'+str(i)+'blur.jpg',blur)
        cv2.imwrite('./res/'+str(i)+'img_candid.jpg',img)
    #cv2.imwrite('./res/'+str(i)+'img.jpg',img)
 

    return candidate




'''

for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if(w > 4*h and w < 10*h):
            crop_img = img_gray[y :y + h , x  :x + w]
            if max < h*w:
                #cv2.drawContours(img, cnt, -1, (0, 255, 0), 3)
                max = h*w
                candidate = cv2.resize(crop_img, (200, 60))
                find = True
max = 0
    candidates = []
    m=0;
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        print(cv2.contourArea(cnt))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        if(w > 2*h and w < 10*h):
            crop_img = img[y :y + h , x  :x + w]
           

            candidate = cv2.resize(crop_img, (200, 60))
            candidates.append(cv2.resize(crop_img, (200, 60)))
            find = True
            cv2.imwrite('./res/'+str(i)+str(m)+'crop_img.jpg',candidate)
            m += 1
            '''