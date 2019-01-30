from skimage import exposure
from skimage import feature
import cv2
import glob
import numpy as np
import imutils
import pickle

CLASS_NUM = 26


def extractFeature(img):

    img = cv2.resize(img, (80,40))
    (H, hogImage) = feature.hog(img, orientations=8, pixels_per_cell=(8, 8),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
		visualize=True)
    hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    hogImage = hogImage.astype("uint8")
	
    #cv2.imwrite('cc.jpg', hogImage)
    
    return H


def svmInit(C=CLASS_NUM, gamma=0.5):
    model = cv2.ml.SVM_create()
    model.setGamma(gamma)
    model.setC(C)
    model.setKernel(cv2.ml.SVM_RBF)
    model.setType(cv2.ml.SVM_C_SVC)
    return model


def train(model, samples, responses):
    
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    return model

def predict(model, samples):
    return model.predict(samples)[1].ravel()

def eval(model, samples, labels):
    predictions = predict(model, samples)
    accuracy = (labels == predictions).mean()
    #print('Accuracy = %.2f %%' % (accuracy * 100))

    confusion = np.zeros((CLASS_NUM, CLASS_NUM), np.int32)
    for i, j in zip(labels, predictions):
        confusion[int(i), int(j)] += 1
    #print('confusion matrix:')
    #print(confusion)


def predict2(model, samples):
    return model.predict(samples)[0].ravel()

# Load from file

def train1():
    train_datas = []
    train_labels = []
    
    

    for i in range (0,26):
        for path in glob.glob('./chars/'+str(i)+'/*.jpg'):
            img =  cv2.imread(path,0)
            img=cv2.resize(img,(20,40))
            #print(img.shape)
            h = extractFeature(img)
            train_datas.append( np.float32(h))
            train_labels.append(i)        

    train_datas = np.squeeze(train_datas)

    print('Training ...')
    model = svmInit()
    train(model, train_datas, np.asarray(train_labels))
    
    print('Evaluating ... ')
    test_datas=[]
    test_labels=[]
    for i in range (0,26):
            for path in glob.glob('./letter/'+str(i)+'/*.jpg'):
                img =  cv2.imread(path,0)
                img=cv2.resize(img,(20,40))
                h = extractFeature(img)
                test_datas.append( np.float32(h))
                test_labels.append(i)
    test_datas = np.squeeze(test_datas)
    eval(model, test_datas, np.asarray(test_labels))
    return model

let=[['b', 10],['t', 11],['j', 12],['d', 13],['s', 14],['ta', 15]
        ,['ai', 16],['g', 17],['sa', 18],['l', 19],['m', 20],['n', 21]
        ,['v', 22],['h', 23],['e', 24],['r', 25]]

def test1(model, img):       
    h = extractFeature(img)
    h=np.float32(h)

    #model.predict(h)[0].ravel()
    l,s=model.predict(np.ravel(h)[None, :])
   
    for l in let:
        if(int(s[0][0]) == l[1]):
            m=l[0]
            return m
    return str(int(s[0][0]))





