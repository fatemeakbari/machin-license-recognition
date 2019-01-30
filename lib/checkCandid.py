import cv2
import numpy as np

import glob

def check(candids):
	
	temps=[]
	for i in glob.glob('./temps/*.jpg'):
		print(i)
		temps.append(cv2.imread(i,0))
	

	method = 'cv2.TM_SQDIFF_NORMED'
	method = eval(method)
	sumList=[]
	i=0;
	for cand in candids:
		sum=0
		for t in temps:
			res = cv2.matchTemplate(cand, t, method)
			min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
			print(min_val)
			sum+=min_val
		sumList.append(sum)
		print('*******************************************')
	

	print(sumList)



