import cv2
import numpy as np
import datetime
img1 = cv2.imread('loa.jpg')
cap = cv2.VideoCapture(1)
wht = 320
confThreshold = 0.5
nmsThreshold = 0.3

flag = True
classesFile = 'obj.names'
classNames = []
with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


modelConfig = 'yolov4-tiny-custom.cfg'
modelWeight = 'yolov4-tiny-custom.weights'

net = cv2.dnn.readNetFromDarknet(modelConfig,modelWeight)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)




def findObjects(outputs,img):

    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nms_threshold=nmsThreshold)
    print(indices)

    for i in indices:
        # global dem
        i = i[0]
        box= bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        timestr = str(now.hour) + ":" + str(now.minute) + ":" + str(now.second)+"_"+str(now.date())
        print(timestr)
        cv2.imwrite('C:/Users/Administrator/Desktop/xu_li_anh/test/%r_%rh%rm%rs_%r.jpg' %(classNames[classIds[i]].upper(),now.hour,now.minute,now.second,now.date()),img)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)

#----------video,webcam
while True:
    now = datetime.datetime.now()
    success, img = cap.read()
    img = cv2.flip(img,1)
    blob = cv2.dnn.blobFromImage(img,1/255,(wht,wht),[0,0,0],1,crop=True)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    outputs= net.forward(outputNames)
    findObjects(outputs,img)
    cv2.imshow('cam',img)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break
cap.release()

#------img
# blob = cv2.dnn.blobFromImage(img1, 1 / 255, (wht, wht), [0, 0, 0], 1, crop=False)
# net.setInput(blob)
#
# layerNames = net.getLayerNames()
# outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# outputs = net.forward(outputNames)
# findObjects(outputs, img1)
#
# cv2.imshow('cam', img1)
# cv2.waitKey(0)
#----------

cv2.destroyAllWindows()