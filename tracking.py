from dis import dis
import math
import cv2
import numpy as np
import detect as dt

model_path="dnn_model\yolov5s.onnx"
classesFile = 'dnn_model\classes.txt' 


detection_threshold = 0.45
nms_threshold = 0.5 # The lower, the more it will supress boxes. Meaning 1 is no supression at all

cap = cv2.VideoCapture('test_video.mp4')
net = cv2.dnn.readNet(model_path)

classes= []
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

center_points_past_frame = []
tracking_objects = {}
track_id = 0
framen = 0
while True:
    ret, frame = cap.read()
    framen += 1
    if not ret:
        break

    dt.setImage(frame, net)
    predictions = dt.detectObjects(net)
    boxes, confidences, classNames = dt.getDetectData(frame, predictions, classes, detection_threshold, nms_threshold)
    dt.draw_boxes(frame, boxes, classNames, confidences, labels=False)


    center_points_frame = []

    for box in boxes:
        (x, y, w, h) = box
        cx = int(x+0.5*w)
        cy = int(y+0.5*h)
        #cv2.circle(frame, (cx,cy), 3, (0,0,255), 2)
        center_points_frame.append((cx,cy))



    if framen <=2:
        for point in center_points_frame:

            for point2 in center_points_past_frame:
                distance = math.dist(point, point2)

                if distance < 20:
                    tracking_objects[track_id] = point
                    track_id += 1
    else:
        for object_id, point2 in tracking_objects.copy().items():
            object_exists = False
            for point in center_points_frame.copy():

                distance = math.dist(point, point2)

                if distance < 20:
                    tracking_objects[object_id] = point   
                    object_exists = True
                    if point in center_points_frame:
                        center_points_frame.remove(point)
                    continue 

            if not object_exists:
                tracking_objects.pop(object_id)


        for point in center_points_frame:
            tracking_objects[track_id] = point
            track_id += 1


    for object_id, point in tracking_objects.items():
        cv2.circle(frame, point, 3, (0,0,255), 2)
        cv2.putText(frame,str(object_id),(point[0],point[1]-7),0,1,(0,0,255),2)

        

    cv2.imshow('frame', frame)
    

    center_points_past_frame = center_points_frame.copy()


    if cv2.waitKey(25) & 0xFF == ord('q'):
      break




cap.release()
cv2.destroyAllWindows()














