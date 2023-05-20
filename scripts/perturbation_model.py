%load_ext autoreload
%autoreload 2

import os
import torch
import torch.nn.functional as F
import numpy as np
import csv
import cv2
import skimage.feature as feature
import torchvision.transforms as transforms
import random

#libraries for yolo
from pytorchyolo.models import load_model
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
from pytorchyolo.utils.utils import non_max_suppression
from pytorchyolo.utils.loss import compute_loss


class yolo():
    def __init__(self):
        self.device, self.model = load_model('./weights/yolo_face_sthanhng.cfg', "./weights/yolo_face_sthanhng.weights")
        self.model.eval()
        for yolo_layer in self.model.yolo_layers:
            yolo_layer.gradient_mode = False

        self.model.gradient_mode = False
    
    def load_image(filename):
        data = cv2.imread(path)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        data = transforms.Compose([DEFAULT_TRANSFORMS,Resize(416)])((data, np.zeros((1, 5))))[0].unsqueeze(0) # transform the image
        return data

    def forward(self, data, conf_thres=0.5, iou_thres=0.5):
        torch.autograd.set_detect_anomaly(True)

        with torch.no_grad():
            nms, nms_output = non_max_suppression(model(data), conf_thres, iou_thres)

        face_list = []
        if type(nms_output[0]) is not int:
            face_list = nms_output[0]

        data = data.to(device)
        data.requires_grad = True

        self.model.gradient_mode = True
        for yolo_layer in self.model.yolo_layers:
            yolo_layer.gradient_mode = True

        return face_list, model(data)
    
    def compute_loss(self, output, target, model):
        return compute_loss(output, target, model)

# TODO replace 416 with model.input_width / model.input_height
def perturb_image(model, filename, mask_type="bbox", mask=None, min_e_model_pth="model.pkl"):
    data = model.load_image
    face_list, output = model.forward(data)
    output_img = data.detach().clone()
    
    for face_index, face_row in enumerate(face_list):
        x, y, w, h = face_row[0], face_row[1], face_row[2], face_row[3]

        factor_x = random.uniform(1, 2)
        factor_y = random.uniform(1, 2)
        factor_w = random.uniform(1, 2)
        factor_h = random.uniform(1, 2)
        
        normal_x, normal_y, normal_w, normal_h = x / 416, y / 416, w / 416, h / 416

        new_x = normal_x * factor_x if random.choice([True, False]) else normal_x / factor_x
        new_y = normal_y * factor_y if random.choice([True, False]) else normal_y / factor_y
        new_w = normal_w * factor_w if random.choice([True, False]) else normal_w / factor_w
        new_h = normal_h * factor_h if random.choice([True, False]) else normal_h / factor_h

        new_x, new_y, new_w, new_h = max(min(1, new_x), 0), max(min(1, new_y), 0), max(min(1, new_w), 0), max(min(1, new_h), 0)

        target = torch.tensor([[0.0, 0, new_x, new_y, new_w, new_h]])
        target = target.to(device)
        
        # Calculate loss with respect to target image
        loss, loss_components = model.compute_loss(output, target, model)

        x1 = max(int(np.floor((x - w / 2).detach().cpu().numpy())), 0)
        y1 = max(int(np.floor((y - h / 2).detach().cpu().numpy())), 0)
        x2 = min(int(np.ceil((x + w / 2).detach().cpu().numpy())), 415)
        y2 = min(int(np.ceil((y + h / 2).detach().cpu().numpy())), 415)

        cropped_image = detach_cpu(data)[:, :, y1:y2, x1:x2]
        cropped_image = tensor_to_image(cropped_image)

        model.zero_grad()
        data.grad = None
        
        loss.backward(retain_graph=True)

        data_grad = data.grad.data
        
        # Generate mask
        bbox = (x1, y1, x2, y2)
        
        mask = np.zeros(data.shape)
        if mask_type == "mask":
            mask[..., y1:y2, x1:x2] = mask
        elif mask_type == "bg":
            mask[..., y1:y2, x1:x2] = (1 - whole_mask[..., y1:y2, x1:x2])
        elif mask_type == "bbox":
            mask[..., y1:y2, x1:x2] = 1
        elif mask_type == "lbbox":
            large_x1 = max(int(np.floor((x - w).detach().cpu().numpy())), 0)
            large_y1 = max(int(np.floor((y - h).detach().cpu().numpy())), 0)
            large_x2 = min(int(np.ceil((x + w).detach().cpu().numpy())), 415)
            large_y2 = min(int(np.ceil((y + h).detach().cpu().numpy())), 415)
            mask[..., large_y1:large_y2, large_x1:large_x2] = 1
            
        joblib.load(min_e_model, min_e_model_pth)
        
        features = get_features()
        
        min_e_pred = min_e_model.pred(features)
        
        sign_data_grad = data_grad.sign()
        output_img = output_img + e * sign_data_grad * mask
        output_img = torch.clamp(output_img, 0, 1)
        
    cv2.imwrite(output_img, "perturbed_" + filename)
    return output_img
