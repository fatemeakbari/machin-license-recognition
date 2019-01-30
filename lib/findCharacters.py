import cv2
import numpy as np
def min_area(w1,h1, w2,h2):
    s1 = w1*h1
    s2 = w2*h2
    if s1 > s2:
        return 1, s2
    elif s2 > s1:
        return 2, s1
    else:
        return 3,s1
    

def overlap(rects):  # returns None if rectangles don't intersect
    loop = len(rects)
    del_list = []
    for i in range(0,loop):
        for j in range(0,loop):
            if(i != j):
                dx = min(rects[i][2], rects[j][2]) - max(rects[i][0], rects[j][0])
                dy = min(rects[i][3], rects[j][3]) - max(rects[i][1], rects[j][1])
                index, s = min_area(rects[i][4],rects[i][5], rects[j][4],rects[j][5])
                
                if(dx*dy >= 0.5*s):
                    if(index==1):
                        del_list.append(rects[j][6])
                    elif(index== 2):
                        del_list.append(rects[i][6])
                    elif(index==3):
                        del_list.append(rects[min(i,j)][6])

    return del_list


def findCharacters(img,dirc):
    edges_img = cv2.Canny(img,100,200)

    (contours, hierarchy) = cv2.findContours(edges_img.astype(np.uint8),
                                             cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   
    
    
    maxArea = max(contours, key = cv2.contourArea)
    minArea = min(contours, key = cv2.contourArea)

    width = img.shape[0]
    height = img.shape[1]


    segments = []
    rects = []
    segnum = 0
    sort=[]
    contours =sorted(contours, key=cv2.contourArea, reverse=True)[:len(contours)]
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        
        if w < 0.2 * height and w > 0.05 * height and h < 0.9*width and  h > 0.5*width :
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            im = img[y:y + h, x:x + w]
            segments.append((x,cv2.resize(im,(20,40))))
            rects.append([x,y,x+w,y+h, w,h , segnum])
            segnum += 1;
    

    
    del_list = overlap(rects)
    del_list = list(set(del_list))
    del_list.sort(reverse= True)
    
    for i in del_list:
        del(segments[i])
    segments.sort(key=lambda x: x[0])
    segs=[]
    
    
    length = min(8,len(segments))
    cv2.imwrite('./res/'+str(i)+'seg.jpg', img)
    for i in range(0,length):
        
        cv2.imwrite('./res/'+str(dirc)+'/'+str(i)+'cnt.jpg', segments[i][1])
        segs.append(segments[i][1])
    
    return(segs)
