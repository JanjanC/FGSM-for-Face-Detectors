import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import mediapipe as mp
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
            
"""
def min_model_eps(image, data_grad, det_fn, start = 0, end = 3):
    # Set epsilon to the start value
    eps = start
    # Increase the epsilon value by 0.05 until it cannot be detected by the detection function or until the end
    sign_data_grad = data_grad.sign()
    bigstep = 0.05 * sign_data_grad
    smallstep = 0.01 * sign_data_grad
    perturbed_img = image.clone().detach()
    yuh = np.moveaxis((perturbed_img.numpy() * 255).squeeze(), 0, -1).astype('uint8')
    cv2.imwrite('yuh.jpg', cv2.cvtColor(yuh, cv2.COLOR_RGB2BGR))
    while det_fn(perturbed_img) and eps < end:
        perturbed_img = perturbed_img + bigstep
        eps += 0.05
        perturbed_img = torch.clamp(perturbed_img, 0, 1)
    yuh = np.moveaxis((perturbed_img.numpy() * 255).squeeze(), 0, -1).astype('uint8')
    cv2.imwrite('yuh1.jpg', cv2.cvtColor(yuh, cv2.COLOR_RGB2BGR))
    # Decrease the epsilon value by 0.01 until it can be detected by the detection function or until the start
    while not det_fn(perturbed_img) and eps > start:
        perturbed_img = perturbed_img - smallstep
        eps -= 0.01
        perturbed_img = torch.clamp(perturbed_img, 0, 1)
    # Add an additional 0.01 so that the returned value is the last epsilon value that the model was unable to detect
    yuh = np.moveaxis((perturbed_img.numpy() * 255).squeeze(), 0, -1).astype('uint8')
    cv2.imwrite('yuh2.jpg', cv2.cvtColor(yuh, cv2.COLOR_RGB2BGR))
    return eps + 0.01
"""

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image
    
def min_model_eps(image, data_grad, det_fn, start = 0., end = 3, step = 0.05):
    # Set epsilon to the start value
    eps = start
    # Increase the epsilon value by 0.05 until it cannot be detected by the detection function or until the end
    perturbed_img = image.clone().detach()
    yuh = np.moveaxis((perturbed_img.numpy() * 255).squeeze(), 0, -1).astype('uint8')
    cv2.imwrite('_1unperturbed.jpg', cv2.cvtColor(yuh, cv2.COLOR_RGB2BGR))
    while det_fn(perturbed_img.detach()) and eps < end:
        eps += 0.05
        perturbed_img = fgsm_attack(image, eps, data_grad)
    yuh = np.moveaxis((perturbed_img.detach().numpy() * 255).squeeze(), 0, -1).astype('uint8')
    cv2.imwrite('_2cantdetect.jpg', cv2.cvtColor(yuh, cv2.COLOR_RGB2BGR))
    # Decrease the epsilon value by 0.01 until it can be detected by the detection function or until the start
    while not det_fn(perturbed_img.detach()) and eps > start:
        eps -= 0.01
        perturbed_img = fgsm_attack(image, eps, data_grad)
    # Add an additional 0.01 so that the returned value is the last epsilon value that the model was unable to detect
    yuh = np.moveaxis((perturbed_img.detach().numpy() * 255).squeeze(), 0, -1).astype('uint8')
    cv2.imwrite('_3maxcandetect.jpg', cv2.cvtColor(yuh, cv2.COLOR_RGB2BGR))
    return eps + 0.01

def mp_det_fn(image):
    image = np.moveaxis((image.numpy() * 255).squeeze(), 0, -1).astype('uint8')
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detection:
        results = face_detection.process(image)
        return results.detections is not None
    
def yn_det_fn(image):
    image = np.moveaxis((image.numpy() * 255).squeeze(), 0, -1).astype('uint8')
    height, width, _ = image.shape
    yn_face_detector.setInputSize((width, height))
    _, faces = yn_face_detector.detect(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return faces is not None
    
# mediapipe stuff
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
    
# yunet stuff
yunet_onnx()
yn_face_detector = cv2.FaceDetectorYN_create("onnx/face_detection_yunet_2022mar.onnx", "", (0, 0))