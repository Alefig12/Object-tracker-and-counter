import numpy as np
import cv2


def setImage(source, net):

    #Resize and normalize 
    result = cv2.dnn.blobFromImage(source, 1/255, (640, 640), [0,0,0],swapRB=True, crop = False)

    #Set image as input for the network
    net.setInput(result)

def detectObjects(net):
    return net.forward(net.getUnconnectedOutLayersNames())

    
    
def drawDetections(img, predicted, classNames, detection_threshold = 0.45, nms_threshold = 0.5):
    
    boxes, confidences, classes_id = getDetectionProperties(img, predicted, detection_threshold)



    # Filter overlaped boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, detection_threshold, nms_threshold)

    # Draw Boxes and labels on image

    draw_boxes(img, boxes, indices, classNames, classes_id, confidences)
    return





def draw_boxes(img, boxes, indices, classNames, classes_id, confidences):

    for i in indices:
        box = boxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        className = classNames[classes_id[i]]
        confidence = confidences[i]
        label = '{}:{:.2f}'.format(className, confidence)
        draw_label(img, label,x,y)

        

        cv2.rectangle(img, (x, y),  (x + w, y + h), (0, 0, 255), thickness=2)

def getDetectionProperties(img, predicted, detection_threshold):

    boxes = []
    confidences = []
    classes_id = []

    #Loop through all detections
    for det in predicted[0][0]:

        confidence = det[4]
        #If confidence bigger than threshold, store class id, confidence and box coordinates
        if confidence >= detection_threshold:
            
            image_height, image_width, _ = img.shape
            
            x_factor = image_width / 640
            y_factor =  image_height / 640
            
            
            x, y, w, h = det[0], det[1], det[2], det[3] #Get coordinates from detection array

            left = int((x - w/2)* x_factor)
            top = int((y -  h/2)* y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            class_id = np.argmax(det[5:]) 

            classes_id.append(class_id)
            boxes.append([left, top, width, height])
            confidences.append(confidence)

    return boxes, confidences, classes_id        
            



def draw_label(img, label, x, y):


    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    dim, baseline = text_size[0], text_size[1]

    cv2.rectangle(img, (x,y), (x + dim[0], y + dim[1] + baseline), (0,0,0), cv2.FILLED)

    cv2.putText(img, label, (x, y + dim[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)


