import os
import time
import glob
import gdown
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import csv
import cv2
import itertools
import matplotlib.pyplot as plt
import skimage.feature as feature
import xlwings as xw
import torchvision.transforms as transforms

import random

#libraries for yolo
from pytorchyolo.models import load_model
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
from pytorchyolo.utils.utils import non_max_suppression
from pytorchyolo.utils.loss import compute_loss

from matplotlib.ticker import (FormatStrFormatter, AutoMinorLocator, FuncFormatter, )

def detach_cpu(image):
    return image.detach().cpu()

# convert 1x3x416x416 to 416x416x3
def reshape_image(image):
    return np.transpose(np.squeeze(image), (1 ,2, 0))

# convert 1x3x416x416 tensor to 416x416x3 numpy image
def tensor_to_image(image):
    return np.transpose(image.detach().cpu().squeeze().numpy(), (1, 2, 0))

def save_tensor_as_image(image, path):
    save_img = cv2.cvtColor(np.moveaxis((image.detach().numpy() * 255).squeeze(), 0, -1).astype('uint8'), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, save_img)






class LocalBinaryPatterns:
  def __init__(self, numPoints, radius):
    self.numPoints = numPoints
    self.radius = radius

  def describe(self, image, eps = 1e-7):
    lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints+3), range=(0, self.numPoints + 2))

    # Normalize the histogram
    hist = hist.astype('float')
    hist /= (hist.sum() + eps)

    return hist, lbp

# From https://medium.com/mlearning-ai/how-to-plot-color-channels-histogram-of-an-image-in-python-using-opencv-40022032e127
# Extracts image's color channel
def extract_color_channel(path, face_index, image, version):
    # BGR Image Color Conversion
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    r, g, b = cv2.split(rgb)

    r *= 255
    g *= 255
    b *= 255

    r_hist = cv2.calcHist(r, [0], None, [26], [0, 256])
    r_hist = r_hist.ravel()
    r_hist = r_hist.astype('float')
    r_hist /= r_hist.sum()

    g_hist = cv2.calcHist([g], [0], None, [26], [0, 256])
    g_hist = g_hist.ravel()
    g_hist = g_hist.astype('float')
    g_hist /= g_hist.sum()

    b_hist = cv2.calcHist([b], [0], None, [26], [0, 256])
    b_hist = b_hist.ravel()
    b_hist = b_hist.astype('float')
    b_hist /= b_hist.sum()

    h, s, v = cv2.split(hsv)

    s *= 255
    v *= 255

    h_hist_HSV = cv2.calcHist([h], [0], None, [36], [0, 361])
    h_hist_HSV = h_hist_HSV.ravel()
    h_hist_HSV = h_hist_HSV.astype('float')
    h_hist_HSV /= h_hist_HSV.sum()

    s_hist_HSV = cv2.calcHist([s], [0], None, [26], [0, 256])
    s_hist_HSV = s_hist_HSV.ravel()
    s_hist_HSV = s_hist_HSV.astype('float')
    s_hist_HSV /= s_hist_HSV.sum()

    v_hist_HSV = cv2.calcHist([v], [0], None, [26], [0, 256])
    v_hist_HSV = v_hist_HSV.ravel()
    v_hist_HSV = v_hist_HSV.astype('float')
    v_hist_HSV /= v_hist_HSV.sum()

    h, l, s = cv2.split(hls)

    l *= 255
    s *= 255

    h_hist_HSL = cv2.calcHist([h], [0], None, [36], [0, 361])
    h_hist_HSL = h_hist_HSL.ravel()
    h_hist_HSL = h_hist_HSL.astype('float')
    h_hist_HSL /= h_hist_HSL.sum()

    l_hist_HSL = cv2.calcHist([l], [0], None, [26], [0, 256])
    l_hist_HSL = l_hist_HSL.ravel()
    l_hist_HSL = l_hist_HSL.astype('float')
    l_hist_HSL /= l_hist_HSL.sum()

    s_hist_HSL = cv2.calcHist([s], [0], None, [26], [0, 256])
    s_hist_HSL = s_hist_HSL.ravel()
    s_hist_HSL = s_hist_HSL.astype('float')
    s_hist_HSL /= s_hist_HSL.sum()

    l, a, b = cv2.split(lab)

    l_hist_LAB = cv2.calcHist([l], [0], None, [26], [0, 256])
    l_hist_LAB = l_hist_LAB.ravel()
    l_hist_LAB = l_hist_LAB.astype('float')
    l_hist_LAB /= l_hist_LAB.sum()

    a_hist_LAB = cv2.calcHist([a], [0], None, [26], [0, 256])
    a_hist_LAB = a_hist_LAB.ravel()
    a_hist_LAB = a_hist_LAB.astype('float')
    a_hist_LAB /= a_hist_LAB.sum()

    b_hist_LAB = cv2.calcHist([b], [0], None, [26], [0, 256])
    b_hist_LAB = b_hist_LAB.ravel()
    b_hist_LAB = b_hist_LAB.astype('float')
    b_hist_LAB /= b_hist_LAB.sum()


    y, cr, cb = cv2.split(ycrcb)

    y *= 255
    cr *= 255
    cb *= 255

    y_hist = cv2.calcHist([y], [0], None, [26], [0, 256])
    y_hist = y_hist.ravel()
    y_hist = y_hist.astype('float')
    y_hist /= y_hist.sum()

    cr_hist = cv2.calcHist([cr], [0], None, [26], [0, 256])
    cr_hist = cr_hist.ravel()
    cr_hist = cr_hist.astype('float')
    cr_hist /= cr_hist.sum()

    cb_hist = cv2.calcHist([cb], [0], None, [26], [0, 256])
    cb_hist = cb_hist.ravel()
    cb_hist = cb_hist.astype('float')
    cb_hist /= cb_hist.sum()

    face_index = str(face_index)

    return r_hist, g_hist, b_hist, h_hist_HSV, s_hist_HSV, v_hist_HSV, h_hist_HSL, s_hist_HSL, l_hist_HSL, l_hist_LAB, a_hist_LAB, b_hist_LAB, y_hist, cr_hist, cb_hist

# From https://medium.com/mlearning-ai/color-shape-and-texture-feature-extraction-using-opencv-cb1feb2dbd73
# Extracts Local Binary Pattern (Texture) of an image
def extract_lbp(path, face_index, image, version):
    # reads the input image as a grayscale image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    desc = LocalBinaryPatterns(24, 8)
    lbp_hist, lbp_img = desc.describe(gray)

    return lbp_hist

# From https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html and https://gist.github.com/rahit/c078cabc0a48f2570028bff397a9e154
def extract_gradients(path, face_index, image, version):
    # Uses the Sobel Filter to extract the gradients of an image
    # reads the input image, then converts BGR color space to RGB
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # compute the 1st order Sobel derivative in X-direction
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)

    # compute the 1st order Sobel derivative in Y-direction
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    # combine sobelx and sobely to form sobel
    sobel = sobelx + sobely

    sobelx_hist = cv2.calcHist([np.float32(sobelx)], [0], None, [26], [0, 256])
    sobelx_hist = sobelx_hist.ravel()
    sobelx_hist = sobelx_hist.astype('float')
    sobelx_hist /= sobelx_hist.sum()

    sobely_hist = cv2.calcHist([np.float32(sobely)], [0], None, [26], [0, 256])
    sobely_hist = sobely_hist.ravel()
    sobely_hist = sobely_hist.astype('float')
    sobely_hist /= sobely_hist.sum()

    sobel_hist = cv2.calcHist([np.float32(sobel)], [0], None, [26], [0, 256])
    sobel_hist = sobel_hist.ravel()
    sobel_hist = sobel_hist.astype('float')
    sobel_hist /= sobel_hist.sum()

    return sobelx_hist, sobely_hist, sobel_hist

def extract_image_attributes(row, path, face_index, image, version):
    r_hist, g_hist, b_hist, h_hist_HSV, s_hist_HSV, v_hist_HSV, h_hist_HSL, s_hist_HSL, l_hist_HSL, l_hist_LAB, a_hist_LAB, b_hist_LAB, y_hist, cr_hist, cb_hist = extract_color_channel(path, face_index, image, version)
    lbp_hist = extract_lbp(path, face_index, image, version)
    sobelx_hist, sobely_hist, sobel_hist = extract_gradients(path, face_index, image, version)

    RGB_size = SV_HSV_size = SL_HSL_size = LAB_size = YCRCB_size = 26

    for i in range(0, RGB_size):
        row['R_BIN_' + version + '_' + str(i)] = r_hist[i]

    for i in range(0, RGB_size):
        row['G_BIN_' + version + '_' + str(i)] = g_hist[i]

    for i in range(0, RGB_size):
        row['B_BIN_' + version + '_' + str(i)] = b_hist[i]

    for i in range(0, h_hist_HSV.size):
        row['H_HSV_BIN_' + version + '_' + str(i)] = h_hist_HSV[i]

    for i in range(0, SV_HSV_size):
        row['S_HSV_BIN_' + version + '_' + str(i)] = s_hist_HSV[i]

    for i in range(0, SV_HSV_size):
        row['V_HSV_BIN_' + version + '_' + str(i)] = v_hist_HSV[i]

    for i in range(0, h_hist_HSL.size):
        row['H_HSL_BIN_' + version + '_' + str(i)] = h_hist_HSL[i]

    for i in range(0, SL_HSL_size):
        row['S_HSL_BIN_' + version + '_' + str(i)] = s_hist_HSL[i]

    for i in range(0, SL_HSL_size):
        row['L_HSL_BIN_' + version + '_' + str(i)] = l_hist_HSL[i]

    for i in range(0, LAB_size):
        row['L_LAB_BIN_' + version + '_' + str(i)] = l_hist_LAB[i]

    for i in range(0, LAB_size):
        row['A_LAB_BIN_' + version + '_' + str(i)] = a_hist_LAB[i]

    for i in range(0, LAB_size):
        row['B_LAB_BIN_' + version + '_' + str(i)] = b_hist_LAB[i]

    for i in range(0, YCRCB_size):
        row['Y_BIN_' + version + '_' + str(i)] = y_hist[i]

    for i in range(0, YCRCB_size):
        row['CR_BIN_' + version + '_' + str(i)] = cr_hist[i]

    for i in range(0, YCRCB_size):
        row['CB_BIN_' + version + '_' + str(i)] = cb_hist[i]

    for i in range(0, lbp_hist.size):
        row["LBP_BIN_" + version + '_' + str(i)] = lbp_hist[i]

    for i in range(0, sobelx_hist.size):
        row["SOBELX_BIN_" + version + '_' + str(i)] = sobelx_hist[i]

    for i in range(0, sobely_hist.size):
        row["SOBELY_BIN_" + version + '_' + str(i)] = sobely_hist[i]

    for i in range(0, sobel_hist.size):
        row["SOBEL_BIN_" + version + '_' + str(i)] = sobel_hist[i]

    return row








OUTPUT_FOLDER = os.path.join(os.getcwd(), "_temp")
RESTORED_MASK_PATH = os.path.join(os.getcwd(), OUTPUT_FOLDER, '_restored_mask')

def load_mask(filename, face_num, target_bbox, bbox, bbox_pad):
    
    filename = "restored_mask_" + os.path.splitext(filename)[0] + "_" + str(face_num) + "_image_final.png"
    mask = cv2.imread(os.path.join(os.getcwd(), RESTORED_MASK_PATH, filename), 0)
    
    x1_pad, y1_pad, x2_pad, y2_pad = bbox_pad
    x1, y1, x2, y2 = bbox
    
    
    padded_dim = (int(x2_pad - x1_pad), int(y2_pad - y1_pad))
    target_dim = (int(target_bbox[2] - target_bbox[0]), int(target_bbox[3] - target_bbox[1]))
    
    if dict(zip(*np.unique(mask, return_counts = True)))[255] < int(target_dim[0] * target_dim[1] * 0.1):
        return torch.ones((1, 3, target_dim[1], target_dim[0])), False
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(mask.shape[0] * 0.5), int(mask.shape[1] * 0.5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    mask = transforms.Compose([DEFAULT_TRANSFORMS])((mask, np.zeros((1, 5))))[0].unsqueeze(0)
    
    #print(os.path.join(os.getcwd(), RESTORED_MASK_PATH, filename))
    #print("mask size before:", mask.shape)
    
    
    #print("x1, y1, orig shape")
    #print(int(face_row["x1_pad"]), int(face_row["y1_pad"]), int(face_row["x2_pad"]), int(face_row["y2_pad"]))
    #print(original_shape)
    
    #print("target shape:", original_shape)
    #print("yoloshape:", int(face_row["x2"] - face_row["x1"]), int(face_row["y2"] - face_row["y1"]))
    
    current_dim = max(mask.shape)
    diff_x, diff_y = abs(padded_dim[0] - current_dim) / 2, abs(padded_dim[1] - current_dim) / 2
    #print("first diff:", diff_x, diff_y)
    
    if diff_y != 0:
        mask = mask[..., int(np.floor(diff_y)):-int(np.ceil(diff_y)), :]
    if diff_x != 0:
        mask = mask[..., int(np.floor(diff_x)):-int(np.ceil(diff_x))]
        
    #print(mask.shape == padded_dim, mask.shape, padded_dim, target_dim)
    
    padding = [
        int(abs(x1 - x1_pad)),
        int(abs(y1 - y1_pad)),
        int(abs(x2 - x2_pad)),
        int(abs(y2 - y2_pad))
    ]
    
    #print("padding:", padding)
    
    new_dim = padded_dim[0] - padding[0] - padding[2], padded_dim[1] - padding[1] - padding[3]
    diff_x, diff_y = (target_dim[0] - new_dim[0]) / 2, (target_dim[1] - new_dim[1]) / 2
    #print("second diff:", diff_x, diff_y)
    
    padding[0] -= int(np.floor(diff_x))
    padding[1] -= int(np.floor(diff_y))
    padding[2] -= int(np.ceil(diff_x))
    padding[3] -= int(np.ceil(diff_y))
    
    #print("mask size after:", mask.shape)
    #print("unpadded bbox:", (face_row["x1"], face_row["y1"], face_row["x2"], face_row["y2"]))
    #print("adjusted padding:", padding)
    # mask = mask[..., padding[1]:-padding[3], padding[0]:-padding[2]]
    
    mask = F.pad(input=mask, pad=(-padding[0], -padding[2], -padding[1], -padding[3]), mode='constant', value=0)
    
    return mask, True









def fgsm_attack(image, e, data_grad, mask):
    # Collect the element-wise sign of the data gradient
    image = image.clone().detach()
    sign_data_grad = data_grad.sign()
    mask = torch.tensor(mask)
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image
    perturbed_image = perturbed_image + e * sign_data_grad * mask
    # apply it only to the face region
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image



device, model = load_model('./weights/yolo_face_sthanhng.cfg', "./weights/yolo_face_sthanhng.weights")
model.eval()
# extract_region - 0 mask, 1 bbox
# perturb_region - 0 - mask, 1 - bbox, 2-lbbox
def pipeline(path, eps_model, color_space, extract_region, perturb_region, given_index=None, model=model, device=device):

    torch.autograd.set_detect_anomaly(True)
    row = {} #the information/columns for a single row in the dataset is stored here

    df = pd.DataFrame() # dataframe storing the dataset
    row['path'] = path
    file_basename = os.path.basename(path)

    model.eval()

    model.gradient_mode = False
    for yolo_layer in model.yolo_layers:
        yolo_layer.gradient_mode = False

    # read and transform the image from the path
    data = cv2.imread(path)  # read the image
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB) #change to rgb
    data = transforms.Compose([DEFAULT_TRANSFORMS,Resize(416)])((data, np.zeros((1, 5))))[0].unsqueeze(0) # transform the image

    output_image = data.clone().detach()

    with torch.no_grad():
        # Forward pass the data through the model and call non max suppression
        nms, nms_output = non_max_suppression(model(data), 0.5, 0.5) #conf_thres and iou_thres = 0.5

    face_list = []
    if type(nms_output[0]) is not int:
        face_list = nms_output[0]

    data = data.to(device)
    # Set requires_grad attribute of tensor. Important for Attack
    data.requires_grad = True

    model.gradient_mode = True
    for yolo_layer in model.yolo_layers:
        yolo_layer.gradient_mode = True

    output = model(data)
    
    # loop through each of the faces in the image
    for face_index, face_row in enumerate(face_list): #nms_output[0] because the model is designed to take in several images at a time from the dataloader but we are only loading the image one at a time
        if given_index == None or given_index == face_index:
            row['face_index'] = face_index
            print("Face", face_index)

            x, y, w, h = face_row[0], face_row[1], face_row[2], face_row[3]
            row['x'], row['y'], row['w'], row['h'] = x, y, w, h

            factor_x, factor_y, factor_w, factor_h = random.uniform(1, 2), random.uniform(1, 2), random.uniform(1, 2), random.uniform(1, 2)
            normal_x, normal_y, normal_w, normal_h = x / 416, y / 416, w / 416, h / 416

            new_x = normal_x * factor_x if random.choice([True, False]) else normal_x / factor_x
            new_y = normal_y * factor_y if random.choice([True, False]) else normal_y / factor_y
            new_w = normal_w * factor_w if random.choice([True, False]) else normal_w / factor_w
            new_h = normal_h * factor_h if random.choice([True, False]) else normal_h / factor_h

            new_x, new_y, new_w, new_h = max(min(1, new_x), 0), max(min(1, new_y), 0), max(min(1, new_w), 0), max(min(1, new_h), 0)

            target = torch.tensor([[0.0, 0, new_x, new_y, new_w, new_h]])
            target = target.to(device)

            loss, loss_components = compute_loss(output, target, model)

            # cropped image with bounding box
            # getting (x1, y1) upper left, (x2, y2) lower right
            x1 = max(int(np.floor((x - w / 2).detach().cpu().numpy())), 0)
            y1 = max(int(np.floor((y - h / 2).detach().cpu().numpy())), 0)
            x2 = min(int(np.ceil((x + w / 2).detach().cpu().numpy())), 415)
            y2 = min(int(np.ceil((y + h / 2).detach().cpu().numpy())), 415)
            
            x1_pad = max(int(np.floor((x - w).detach().cpu().numpy())), 0) # prevent negative values
            y1_pad = max(int(np.floor((y - h).detach().cpu().numpy())), 0)
            x2_pad = min(int(np.ceil((x + w).detach().cpu().numpy())), 415) # prevent from getting out of range
            y2_pad = min(int(np.ceil((y + h).detach().cpu().numpy())), 415)
            
            large_x1 = max(int(np.floor((x - w).detach().cpu().numpy())), 0)
            large_y1 = max(int(np.floor((y - h).detach().cpu().numpy())), 0)
            large_x2 = min(int(np.ceil((x + w).detach().cpu().numpy())), 415)
            large_y2 = min(int(np.ceil((y + h).detach().cpu().numpy())), 415)


            cropped_image = detach_cpu(data)[:, :, y1:y2, x1:x2] #get the first dimension, the channels, and crop it
            cropped_image = tensor_to_image(cropped_image) #reshape the image to (w/h, h/w, channel)

            # Zero all existing gradients
            model.zero_grad()
            data.grad = None

            # Calculate gradients of model in backward pass
            loss.backward(retain_graph=True) #TODO: Amos - check if this is correct

            # Collect datagrad
            data_grad = data.grad.data
            
            bbox = (x1, y1, x2, y2)
            pad_bbox = (x1_pad, y1_pad, x2_pad, y2_pad)
            mask, _ = load_mask(os.path.basename(path), face_index, bbox, bbox, pad_bbox)
            
            if extract_region == "mask":
                row = extract_image_attributes(row, path, face_index, cropped_image * tensor_to_image(mask[0]), "mask")
            else:
                row = extract_image_attributes(row, path, face_index, cropped_image, "bbox")
            
            
            row['x'], row['y'], row['w'], row['h'] = row['x'] / 416, row['y'] / 416, row['w'] / 416, row['h'] / 416
            
            df = df.append(row, ignore_index=True) #append the attributes of one face to the dataframe

            predict_features = get_features(color_space, extract_region)
            X_features = df.loc[:,  predict_features]

            min_eps = eps_model.predict(X_features)

            if perturb_region == 0:
                whole_mask = np.zeros(data.shape)
                whole_mask[..., y1:y2, x1:x2] = mask
                select_mask = whole_mask
            elif perturb_region == 1:
                bbox_mask = np.zeros(data.shape)
                bbox_mask[..., y1:y2, x1:x2] = 1
                select_mask = bbox_mask
            elif perturb_region == 2:
                large_bbox_mask = np.zeros(data.shape)
                large_bbox_mask[..., large_y1:large_y2, large_x1:large_x2] = 1
                select_mask = large_bbox_mask

                
            output_image = fgsm_attack(output_image, min_eps[0], data_grad.clone().detach(), select_mask)

    return output_image










color_channels = {
    "RGB": ("R_BIN_", "G_BIN_", "B_BIN_"),
    "HSV": ("H_HSV_BIN_", "S_HSV_BIN_", "V_HSV_BIN_"),
    "HSL": ("H_HSL_BIN_", "S_HSL_BIN_", "L_HSL_BIN_"),
    "LAB": ("L_LAB_BIN_", "A_LAB_BIN_", "B_LAB_BIN_"),
    "YCBCR": ("Y_BIN_", "CR_BIN_", "CB_BIN_"),
}

def get_features(color_space, region):
    features = ["w", "h", "x", "y"]
    for color_channel in color_channels[color_space]: 
        features += [color_channel + region + "_" + str(i) for i in range(26)]
    features += ["LBP_BIN_" + region + "_" + str(i) for i in range(26)]
    features += ["SOBELX_BIN_" + region + "_" + str(i) for i in range(20)]
    features += ["SOBELY_BIN_" + region + "_" + str(i) for i in range(20)]
    features += ["SOBEL_BIN_" + region + "_" + str(i) for i in range(20)]
    return features