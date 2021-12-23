from cv2 import cv2
import numpy as np
from keras.models import load_model

model = 'model.h5'
image = 'pin2.jpg'

def prediction(image,model):
    a=0
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray,(1,1),0)
    ret,thresh1 = cv2.threshold(gray,175,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    try: hierarchy = hierarchy[0]
    except: hierarchy = []
    for contour, hier in zip(contours, hierarchy):
        area = cv2.contourArea(contour)
        if area >150 and area < 900:
           rect = cv2.minAreaRect(contour)
           box = cv2.boxPoints(rect)
           box = np.int0(box)
           cv2.drawContours(img,[box],0,(0,255,0),2)
           a+=1
    print('so box duoc tao: ',a)
    imgcrop = img[box[2][1]:box[0][1],box[0][0]:box[2][0]]
    imgcrop = cv2.cvtColor(imgcrop,cv2.COLOR_BGR2GRAY)
    imgcrop = cv2.resize(imgcrop,(24,40))
    class_names = ['Cell', 'Diode', 'Diode-Multi', 'No-anomaly', 'Offline-module', 'Shadowing']
    model = load_model(model)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    imgcopy = imgcrop / 255
    imgcopy = np.reshape(imgcopy, [1, 40, 24, 1])
    clas = model.predict(imgcopy)
    classes = np.argmax(model.predict(imgcopy), axis=-1)
    names = [class_names[i] for i in classes]
    accuracy = np.amax(clas) * 100
    imgresize = cv2.resize(imgcrop, (240, 400))
    cv2.putText(imgresize, str(names), (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(imgresize, str(round(accuracy, 2)) + " %", (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow('img', imgresize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

prediction(image,model)
