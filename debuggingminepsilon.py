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
    #print(bboxes, target_bbox)
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
def fgsm_attack(image, e, data_grad, mask):
    # Collect the element-wise sign of the data gradient
    image = image.clone().detach()
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image
    """
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        if mask is not None:
            perturbed_image[..., y1:y2, x1:x2] = perturbed_image[..., y1:y2, x1:x2] + e * sign_data_grad[..., y1:y2, x1:x2] * mask 
        else:
            perturbed_image[..., y1:y2, x1:x2] = perturbed_image[..., y1:y2, x1:x2] + e * sign_data_grad[..., y1:y2, x1:x2]
    else:
        perturbed_image = perturbed_image + e * sign_data_grad
    """
    perturbed_image = perturbed_image + e * sign_data_grad * mask
    # apply it only to the face region
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image
    
def min_model_eps(image, data_grad, det_fn, mask, bbox, start = 0., end = 3, step = 0.05):
    # Set epsilon to the start value
    eps = start
    
    #print("\tbefore perturbation | closest bbox:", closest_bbox(det_fn(image), bbox), "eps:", eps)
    
    # If the current face cannot be detected
    _, iou = closest_bbox(det_fn(image), bbox)
    #print("IOU:", iou)
    if iou <= 0.40:
        return 0
    
    perturbed_img = image.clone().detach()
    
    """ # Save img sample
    save_img = cv2.cvtColor(np.moveaxis((perturbed_img.detach().numpy() * 255).squeeze(), 0, -1).astype('uint8'), cv2.COLOR_RGB2BGR)
    cv2.imwrite('_1unperturbed.png', save_img)
    """
    
    # Increase the epsilon value by 0.05 until it cannot be detected by the detection function or until the end
    while closest_bbox(det_fn(perturbed_img), bbox)[1] > 0.3 and eps < end:
        step = 0.025 if eps < 1 else 0.05
        eps += step
        perturbed_img = fgsm_attack(image, eps, data_grad, mask)
    
    """ # Save img sample
    save_img = cv2.cvtColor(np.moveaxis((perturbed_img.detach().numpy() * 255).squeeze(), 0, -1).astype('uint8'), cv2.COLOR_RGB2BGR)
    cv2.imwrite('_2cantdetect.png', save_img)
    print("\te undetectable | closest bbox:", closest_bbox(det_fn(perturbed_img), bbox), "eps:", eps)
    """
    print("IOU2:", closest_bbox(det_fn(perturbed_img), bbox))
    
    # Decrease the epsilon value by 0.01 until it can be detected by the detection function or until the start
    while not closest_bbox(det_fn(perturbed_img), bbox)[1] > 0.3 and eps > start:
        step = 0.005 if eps < 1 else 0.01
        eps -= step
        perturbed_img = fgsm_attack(image, eps, data_grad, mask)
        
    print("IOU3:", closest_bbox(det_fn(perturbed_img), bbox))
        
    #print(np.array_equal(cv2.imread("_2cantdetect.png"), save_img))
    
    """ # Save img sample
    save_img = cv2.cvtColor(np.moveaxis((perturbed_img.detach().numpy() * 255).squeeze(), 0, -1).astype('uint8'), cv2.COLOR_RGB2BGR)
    cv2.imwrite('_3maxcandetect.png', save_img)
    print("\tmax e detectable | closest bbox:", closest_bbox(det_fn(perturbed_img), bbox), "eps:", eps)
    """
    
    # Add an additional 0.01 so that the returned value is the last epsilon value that the model was unable to detect
    return eps + step

# MediPipe detection function, accepts pytorch tensors returns bounding boxes (x1, y1, x2, y2)
def mp_det_fn(image, return_boxes = True):
    image = np.moveaxis((image.detach().numpy() * 255).squeeze(), 0, -1).astype('uint8')
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detection:
        results = face_detection.process(image)
        if not return_boxes:
            return results.detections is not None
        else:
            if results.detections is None:
                return []
            #print("MP detected n faces:", len(results.detections))
            bboxes = []
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                height, width, _ = image.shape
                #print(np.clip(bbox.xmin, 0., 1.), np.clip(bbox.xmin, 0., 1.))
                #print(bbox.xmin + bbox.width, bbox.ymin + bbox.height, np.clip(bbox.xmin + bbox.width, 0, 1), np.clip(bbox.ymin + bbox.height, 0, 1))
                bboxes += [tuple((
                    *mp_drawing._normalized_to_pixel_coordinates(np.clip(bbox.xmin, 0., 1.), np.clip(bbox.ymin, 0., 1.), width, height),
                    *mp_drawing._normalized_to_pixel_coordinates(np.clip(bbox.xmin + bbox.width, 0., 1.), np.clip(bbox.ymin + bbox.height, 0., 1.), width, height)
                ))]
                
            return bboxes

# YuNet detection function, accepts pytorch tensors returns bounding boxes (x1, y1, x2, y2)
def yn_det_fn(image, return_boxes = True):
    image = np.moveaxis((image.detach().numpy() * 255).squeeze(), 0, -1).astype('uint8')
    height, width, _ = image.shape
    yn_face_detector.setInputSize((width, height))
    yn_face_detector.setNMSThreshold(0.5)
    yn_face_detector.setScoreThreshold(0.5)
    _, faces = yn_face_detector.detect(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not return_boxes:
        return faces is not None
    else:
        if faces is None:
            return []
        #print("YN detected n faces:", len(faces))
        bboxes = []
        for face in faces:
            bboxes += [tuple((
                int(np.floor(face[0])),
                int(np.floor(face[1])),
                int(np.ceil(face[0] + face[2])),
                int(np.ceil(face[1] + face[3]))
            ))]
        return bboxes

# YoloFace detection function, accepts pytorch tensors returns bounding boxes (x1, y1, x2, y2)
def yf_det_fn(image, return_boxes = True):
    image = np.moveaxis((image.detach().numpy() * 255).squeeze(), 0, -1).astype('uint8')
    bboxes = detect.detect_image(yf_face_detector, image)
    if not return_boxes:
        return bboxes is not None
    else:
        return [tuple(map(int, bbox[:4])) for bbox in bboxes]
    
# MediaPipe detector init
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
    
# YuNet detector init
yunet_onnx()
yn_face_detector = cv2.FaceDetectorYN_create("onnx/face_detection_yunet_2022mar.onnx", "", (0, 0))

# YoloFace detector init
_, yf_face_detector = models.load_model('./weights/yolo_face_sthanhng.cfg', "./weights/yolo_face_sthanhng.weights")