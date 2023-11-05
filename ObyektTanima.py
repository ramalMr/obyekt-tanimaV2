import cv2

thres = 0.45  
cap = cv2.VideoCapture(0)  
cap.set(3, 1280)  
cap.set(4, 720)   
cap.set(10, 70)   

classNames = []
classFile = 'adlar.names'  
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'obyektlerpbt.pbtxt'  
weightsPath = 'netice.pb'  

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


cv2.namedWindow("Output", cv2.WINDOW_NORMAL)  
cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId-1].upper(), (box[0]+10, box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0]+200, box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Output", img)  
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()
