import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import torch

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
    perturbed_image = perturbed_image + e * sign_data_grad * mask
    # apply it only to the face region
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image
    
def min_model_eps(image, data_grad, model, mask, bbox, start = 0., end = 3, background=False):
    # Set epsilon to the start value
    eps = start
    
    #print("\tbefore perturbation | closest bbox:", closest_bbox(det_fn(image), bbox), "eps:", eps)
    
    # If the current face cannot be detected
    _, iou = closest_bbox(model.detect(image), bbox)
    #print("IOU:", iou)
    if iou <= 0.40:
        return 0
    
    perturbed_img = image.clone().detach()
    
    #utils.save_tensor_img(perturbed_img, '_1unperturbed.png')
    
    # Increase the epsilon value by 0.05 until it cannot be detected by the detection function or until the end
    while closest_bbox(model.detect(perturbed_img), bbox)[1] > 0.3 and eps < end:
        if background:
            step = 0.5
        else:
            step = 0.025 if eps < 1 else 0.05
        eps += step
        perturbed_img = fgsm_attack(image, eps, data_grad, mask)
    
    if save_image:
        utils.save_tensor_img(perturbed_img, '_1unperturbed.png')y
    
    #utils.save_tensor_img(perturbed_img, '_2cantdetect.png')
    #print("\te undetectable | closest bbox:", closest_bbox(model.detect(perturbed_img), bbox), "eps:", eps)
    #print("IOU2:", closest_bbox(model.detect(perturbed_img), bbox)[1])
    
    # Decrease the epsilon value by 0.01 until it can be detected by the detection function or until the start
    while not closest_bbox(model.detect(perturbed_img), bbox)[1] > 0.3 and eps > start:
        step = 0.005 if eps < 1 else 0.01
        eps -= step
        perturbed_img = fgsm_attack(image, eps, data_grad, mask)
        
    #print("IOU3:", closest_bbox(model.detect(perturbed_img), bbox)[1])
        
    #print(np.array_equal(cv2.imread("_2cantdetect.png"), save_img))
    
    #utils.save_tensor_img(perturbed_img, '_3maxcandetect.png')
    #print("\tmax e detectable | closest bbox:", closest_bbox(model.detect(perturbed_img), bbox), "eps:", eps)
    
    # Add an additional 0.01 so that the returned value is the last epsilon value that the model was unable to detect
    return eps + step

def binary_search(low, high, image, data_grad, model.detect, mask, bbox, background=False):
    _, iou = closest_bbox(model.detect(image), bbox)
    if iou <= 0.4:
        return 0
    
    perturbed_img = image.clone().detach()
    mid = low
 
    while low <= high:
        mid = (high + low) / 2
        perturbed_img = fgsm_attack(image, mid, data_grad, mask)
        iou_score = closest_bbox(model.detect(perturbed_img), bbox)[1]
        if iou_score > 0.3:
            low = mid + 0.005
            mid += 0.005
        else:
            high = mid - 0.005
            
    return mid