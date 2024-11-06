import cv2 #imports openCV 2.0 library.

classNames = [] #defines classNames variable as list
classFile = "/home/pi/Desktop/Object_Detection_Files/coco.names" #THIS REQUIRES THE OBJECT DETECTION FOLDER TO BE IN THE DESKTOP OF THE RPI. MUST FOLLOW CORRECT DIRECTORY
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n") #reads object classifications and names and stores in a list classNames

configPath = "/home/pi/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt" #DEFINES COCO DATASET DIRECTORY
weightsPath = "/home/pi/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath) #setting parameters for DNN
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=[]): #function for classifying objects
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames #if no objects defined, then objects list = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    return img,objectInfo


if __name__ == "__main__":

    cap = cv2.VideoCapture(0) #opens camera module through cv2.
    cap.set(3,640)
    cap.set(4,480)
    #cap.set(10,70)


    while True:
        success, img = cap.read() 
        result, objectInfo = getObjects(img,0.45,0.2) # objects=[] argument allows for specific object tagging.
        #print(objectInfo)
        cv2.imshow("Output",img)
        cv2.waitKey(1)
