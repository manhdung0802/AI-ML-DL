from cv2 import cv2
import numpy as np
# from keras.models import load_model

model = 'model.h5'
image = 'solar-panel5.jpg'

def prediction(image,model):
    list=[]
    a=0
    img = cv2.imread(image)
    img = cv2.resize(img,(900,900))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray,(1,1),0)
    ret,thresh1 = cv2.threshold(gray,94,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    try: hierarchy = hierarchy[0]
    except: hierarchy = []
    for contour, hier in zip(contours, hierarchy):
        area = cv2.contourArea(contour)

        if area >1300 and area < 6000:
           # rect = cv2.minAreaRect(contour)
           # box = cv2.boxPoints(rect)
           # box = np.int0(box)
           # posx= (box[0][0]-box[2][0])/2
           # posy = (box[2][1]-box[1][1])/2
           # cv2.drawContours(img,[box],0,(0,255,0),2)
           # imgcrop = img[box[2][1]:box[0][1], box[0][0]:box[2][0]]
           # # cv2.putText(imgcrop,'1',(4,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,55,255),2)
           # cv2.putText(img, '1', (box[0][0], box[0][1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
           # list.append(imgcrop)
           cv2.drawContours(img, contour, 0, (0, 255, 0), 2)
           x, y, w, h = cv2.boundingRect(contour)
           cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
           cv2.putText(img, '1', (x+20, y+20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2, cv2.LINE_AA)

    # print(box)
    # imgcrop = img[box[2][1]:box[0][1],box[0][0]:box[2][0]]
    # imgcrop = cv2.cvtColor(imgcrop,cv2.COLOR_BGR2GRAY)
    # imgcrop = cv2.resize(imgcrop,(24,40))
    cv2.imshow('stock',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

prediction(image,model)
