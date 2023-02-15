import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import mediapipe as mp
from pytorchyolo import detect, models
import os
import gdown 

def yunet_onnx():
    model_file=[
        'face_detection_yunet_2022mar.onnx'
    ]
    
    gdrive_url=[
        'https://drive.google.com/uc?id=1V7FdMzjwyGPn5QwW18D4cirzbZfKwC-i'
    ]
    
    cwd=os.getcwd() 
    if 'onnx' in os.listdir(cwd):
        for i in range(len(model_file)):
            if model_file[i] in os.listdir(os.path.join(cwd, 'onnx')):
                print(model_file[i] + ':: status : file already exists')
            else:
                gdown.download(gdrive_url[i],os.path.join(cwd, 'onnx', model_file[i]), quiet=False)
    else:
        os.makedirs(os.path.join(cwd,'onnx'))
        for i in range(len(model_file)):
            gdown.download(gdrive_url[i], os.path.join(cwd, 'onnx', model_file[i]), quiet=False)  

def get_iou(ground_truth, pred):
    # coordinates of the area of intersection.
    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[2], pred[2])
    iy2 = np.minimum(ground_truth[3], pred[3])
     
    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
     
    area_of_intersection = i_height * i_width
     
    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1
     
    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1
     
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
     
    iou = area_of_intersection / area_of_union
     
    return iou

def closest_bbox(bboxes, target_bbox):
    ious = []
    for bbox in bboxes:
        iou = get_iou(bbox, target_bbox)
        if iou == 1:
            return bbox, 1
        ious += [iou]
    if ious:
        return bboxes[ious.index(max(ious))], max(ious)
    else:
        return [], 0

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad, x1, y1, x2, y2):
    # Collect the element-wise sign of the data gradient
    image = image.clone().detach()
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image
    perturbed_image[:, :, y1:y2, x1:x2] = perturbed_image[:, :, y1:y2, x1:x2] + epsilon * sign_data_grad[:, :, y1:y2, x1:x2] # apply it only to the face region
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image
    
def min_model_eps(image, data_grad, det_fn, bbox, start = 0., end = 3, step = 0.05):
    # Set epsilon to the start value
    eps = start
    print("\tbefore perturbation | closest bbox:", closest_bbox(det_fn(image), bbox), "eps:", eps)
    bbox, iou = closest_bbox(det_fn(image), bbox)
    if iou <= 0.5:
        return 0
    perturbed_img = image.clone().detach()
    
    save_img = np.moveaxis((perturbed_img.numpy() * 255).squeeze(), 0, -1).astype('uint8')
    cv2.imwrite('_1unperturbed.jpg', cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))
    
    # Increase the epsilon value by 0.05 until it cannot be detected by the detection function or until the end
    while closest_bbox(det_fn(perturbed_img), bbox)[1] > 0.5 and eps < end:
        eps += 0.05
        perturbed_img = fgsm_attack(image, eps, data_grad, *bbox)
    
    save_img = np.moveaxis((perturbed_img.detach().numpy() * 255).squeeze(), 0, -1).astype('uint8')
    cv2.imwrite('_2cantdetect.jpg', cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))
    print("\te undetectable | closest bbox:", closest_bbox(det_fn(perturbed_img), bbox), "eps:", eps)
    
    # Decrease the epsilon value by 0.01 until it can be detected by the detection function or until the start
    while not closest_bbox(det_fn(perturbed_img), bbox)[1] > 0.5 and eps > start:
        eps -= 0.01
        perturbed_img = fgsm_attack(image, eps, data_grad, *bbox)
    
    save_img = np.moveaxis((perturbed_img.detach().numpy() * 255).squeeze(), 0, -1).astype('uint8')
    cv2.imwrite('_3maxcandetect.jpg', cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))
    print("\tmax e detectable | closest bbox:", closest_bbox(det_fn(perturbed_img), bbox), "eps:", eps)
    
    # Add an additional 0.01 so that the returned value is the last epsilon value that the model was unable to detect
    return eps + 0.01

def mp_det_fn(image, return_boxes = True):
    image = np.moveaxis((image.detach().numpy() * 255).squeeze(), 0, -1).astype('uint8')
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detection:
        results = face_detection.process(image)
        if not return_boxes:
            return results.detections is not None
        else:
            if results.detections is None:
                return []
            bboxes = []
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                height, width, _ = image.shape
                bboxes += [tuple((
                    *mp_drawing._normalized_to_pixel_coordinates(bbox.xmin, bbox.ymin, width, height),
                    *mp_drawing._normalized_to_pixel_coordinates(bbox.xmin + bbox.width, bbox.ymin + bbox.height, width, height)
                ))]
                
            return bboxes
    
def yn_det_fn(image, return_boxes = True):
    image = np.moveaxis((image.detach().numpy() * 255).squeeze(), 0, -1).astype('uint8')
    height, width, _ = image.shape
    yn_face_detector.setInputSize((width, height))
    _, faces = yn_face_detector.detect(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not return_boxes:
        return faces is not None
    else:
        if faces is None:
            return []
        bboxes = []
        for face in faces:
            bboxes += [tuple((
                int(np.floor(face[0])),
                int(np.floor(face[1])),
                int(np.ceil(face[0] + face[2])),
                int(np.ceil(face[1] + face[3]))
            ))]
        return bboxes

def yf_det_fn(image, return_boxes = True):
    bboxes = detect.detect_image(yn_face_detector, image)
    if not return_boxes:
        return bboxes is not None
    else:
        return bboxes
    
# mediapipe stuff
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
    
# yunet stuff
yunet_onnx()
yn_face_detector = cv2.FaceDetectorYN_create("onnx/face_detection_yunet_2022mar.onnx", "", (0, 0))

_, yf_face_detector = models.load_model('./weights/yolo_face_sthanhng.cfg', "./weights/yolo_face_sthanhng.weights")