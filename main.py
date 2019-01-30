from lib import *
import cv2

model = train1()

for i in range(0,30):
	segs=[]
	plat =''
	path= './dataset/'+str(i)+'.jpg'
	img = cv2.imread(path)
	candids = findCandidate(img,i)
	if len(candids) != 0:
		candid = candids[0]
		segs = findCharacters(candid,i)
		for seg in segs:
			p=test1(model,seg)
			plat +=(p)
		print(plat)
		im = cv2.putText(img,plat, (200,200), cv2.FONT_HERSHEY_SIMPLEX, 2,2, 3)
		im = cv2.resize(im, (int((400 / img.shape[0]) * img.shape[1]), 400))
		cv2.imshow('image',im)
		cv2.waitKey(5000)

