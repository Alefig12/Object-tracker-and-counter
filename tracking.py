import cv2
import numpy as np
import detect as dt

model_path="dnn_model\yolov5s.onnx"
classesFile = 'dnn_model\classes.txt' 


detection_threshold = 0.45
nms_threshold = 0.5 # The lower, the more it will supress boxes. Meaning 1 is no supression at all


net = cv2.dnn.readNet(model_path)
img = cv2.imread('test.jpeg')

dt.setImage(img, net)


predictions = dt.detectObjects(net)

classNames= []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


dt.drawDetections(img, predictions, classNames, detection_threshold, nms_threshold)


cv2.imshow('img',img)
cv2.waitKey(0)

