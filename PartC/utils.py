# Importing necessary libraries
import time
import random

import numpy as np 
import cv2


def load_image(img_path):
    """Loads and image from path

    Args:
        img_path (str): path of image

    Returns:
        tuple: image, height, width
    """
    img = cv2.imread(img_path)
    h, w, _  = img.shape

    return img, h, w


def yolov3(weights, cfg, classes_fpath):
    """Loads the yolov3 model from given weights and config

    Args:
        weights (str): path of the weights file
        cfg (str): path of the config file
        classes_fpath (str): path of the file with classes

    Returns:
        tuple: model, classes and layers
    """

    # loading the model from architecture and weights
    model = cv2.dnn.readNet(weights, cfg, classes_fpath)
    
    f = open(classes_fpath)
    classes = f.read().strip().split("\n")

    layers = model.getLayerNames()
    output_layers =  [layers[l[0]-1]  for l in model.getUnconnectedOutLayers()]

    return model, classes, output_layers


def get_bounding_boxes(model, img, output_layers, w, h, threshold):
    """Run the model on the bounding boxes and get coordinates of the corner points

    Args:
        model (class): the trained model read using cv2.dnn.readNet()
        img (np.ndarray): input frame
        output_layers (list): layers of the model
        w (int): width of the frame
        h (int): height of the frame
        threshold (float): confidence threshold to be crossed for object detection

    Returns:
        tuple: bounding boxes, confidences and labels
    """

    # preprocessing step
    blob = cv2.dnn.blobFromImage(img, 1./255, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)

    # forward pass or prediciting using the model
    outputs = model.forward(output_layers)

    ## We have to draw bounding boxes around the objects if the confidence level is beyond a certain threshold ##
    boxes = []
    confidences = []
    labels = []

    for output in outputs:
        for params in output:

            class_confidences = params[5:]
            
            label = np.argmax(class_confidences)
            confidence = class_confidences[label]

            if  confidence > threshold:

                c_x, c_y, w_box, h_box = list(map(int,params[:4] * [w, h, w, h]))

                # coordinates of the left-bottom corner of the box
                p_x = int(c_x - w_box/2)
                p_y = int(c_y - h_box/2)

                boxes.append([p_x, p_y, w_box, h_box])
                confidences.append(float(confidence))
                labels.append(label)

    return boxes, confidences, labels


def draw_bounding_boxes(boxes, confidences, labels, classes, img, confidence_threshold, nms_threshold):
    """Draws the given bounding boxes on the frame after detection

    Args:
        boxes (list): bounding boxes
        confidences (list): confidences of each bounding box
        labels (list): label for each object in bounding box
        classes (list): name of classes as a list
        img (np.ndarray): input frame 
        confidence_threshold (float): threshold for confidence
        nms_threshold (float): threshold for nms (prevents overlapping bounding boxes)
    """

    box_idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    if len(box_idxs) > 0:
        for i in box_idxs.flatten():
            x, y, w, h = boxes[i]

            r = lambda: random.randint(0, 255) # generates random colors
            color = (r(), r(), r())
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            description = f"{classes[labels[i]]}: {confidences[i]}"
            cv2.putText(img, description, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    cv2.imshow("Mask detection voohoo!!", img)


def detect_in_video(video_path, webcam, weights, cfg, classes_fpath, confidence_threshold, nms_threshold):
    """Performs mask detection in a video

    Args:
        video_path (str): path of video file if saved video is to be taken as input
        webcam (bool): True if use webcam feed as input else False
        weights (str): path to the file containing saved weights
        cfg (str): path to the config file for the model
        classes_fpath (str): path to the classes file
        confidence_threshold (float): threshold for confidence 
        nms_threshold (float): threshold for nms
    """

    # load the model
    model, classes, output_layers = yolov3(weights, cfg, classes_fpath)

    if webcam: # if webcam feed is to be taken as input
        video = cv2.VideoCapture(0)
        time.sleep(1.0)
    else: # if a saved video is given as input
        video = cv2.VideoCapture(video_path)

    while True:
        ret, img = video.read()

        h, w, _ = img.shape

        boxes, confidences, labels = get_bounding_boxes(model, img, output_layers, w, h, confidence_threshold)  
        draw_bounding_boxes(boxes, confidences, labels, classes, img, confidence_threshold, nms_threshold)

        # quit on pressing q key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()


def detect_in_image(img_path, weights, cfg, classes_fpath, confidence_threshold, nms_threshold):
    """Performs mask detection in an image

    Args:
        img_path (str): path of the image
        weights (str): path of the file storing model weights
        cfg (str): path of the config file for the model
        classes_fpath (str): path of the file containing class labels
        confidence_threshold (float): threshold for confidence
        nms_threshold (float): threshold for nms
    """

    img, h, w = load_image(img_path)
    model, classes, output_layers = yolov3(weights, cfg, classes_fpath)

    boxes, confidences, labels = get_bounding_boxes(model, img, output_layers, w, h, confidence_threshold)
    draw_bounding_boxes(boxes, confidences, labels, classes, img, confidence_threshold, nms_threshold)

    cv2.waitKey(0)
    cv2.destroyAllWindows()










