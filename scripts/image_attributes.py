import pandas as pd
import torch
from scripts.utils import *
from scripts.face_detectors import YoloFace
from pytorchyolo.models import load_model
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
from pytorchyolo.utils.utils import non_max_suppression
from pytorchyolo.utils.loss import compute_loss
import cv2
import os
import numpy as np
import skimage.feature as feature
from scripts.facesegmentor import FaceSegementor

save_color_images = True
save_lbp_images = True
save_gradient_images = True 
FOLDER_PATH = os.path.join(os.getcwd(), "image_attribute_dumps")
FOLDER_NAME = ""

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
    
    if save_color_images:
        COLOR_PATH = os.path.join(FOLDER_PATH, FOLDER_NAME + '_Colors_' + version)
        if not os.path.exists(COLOR_PATH):
            os.makedirs(COLOR_PATH)
            
        cv2.imwrite(os.path.join(COLOR_PATH, os.path.splitext(os.path.basename(path))[0]) + '_RGB_' + version + '_' + str(face_index) + '.png', rgb)
        cv2.imwrite(os.path.join(COLOR_PATH, os.path.splitext(os.path.basename(path))[0]) + '_HSV_' + version + '_' + str(face_index) + '.png', hsv)
        cv2.imwrite(os.path.join(COLOR_PATH, os.path.splitext(os.path.basename(path))[0]) + '_HSL_' + version + '_' + str(face_index) + '.png', hls)
        cv2.imwrite(os.path.join(COLOR_PATH, os.path.splitext(os.path.basename(path))[0]) + '_LAB_' + version + '_' + str(face_index) + '.png', lab)
        cv2.imwrite(os.path.join(COLOR_PATH, os.path.splitext(os.path.basename(path))[0]) + '_YCRCB_' + version + '_' + str(face_index) + '.png', ycrcb)

#     # RGB Image Histogram
#     red_hist = cv2.calcHist([rgb], [0], None, [256], [0, 256])
#     green_hist = cv2.calcHist([rgb], [1], None, [256], [0, 256])
#     blue_hist = cv2.calcHist([rgb], [2], None, [256], [0, 256])

#     # HSV Image Histogram
#     hue_hist_HSV = cv2.calcHist([hsv], [0], None, [256], [0, 256])
#     saturation_hist_HSV = cv2.calcHist([hsv], [1], None, [256], [0, 256])
#     value_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])

#     # HLS Image Histogram
#     hue_hist_HLS = cv2.calcHist([hls], [0], None, [256], [0, 256])
#     lightness_hist_HLS = cv2.calcHist([hls], [1], None, [256], [0, 256])
#     saturation_hist_HLS = cv2.calcHist([hls], [2], None, [256], [0, 256])

#     # LAB Image Histogram
#     lightness_hist_LAB = cv2.calcHist([lab], [0], None, [256], [0, 256])
#     a_hist_LAB = cv2.calcHist([lab], [1], None, [256], [0, 256])
#     b_hist_LAB = cv2.calcHist([lab], [2], None, [256], [0, 256])

#     # YCrCb Image Histogram
#     y_hist = cv2.calcHist([ycrcb], [0], None, [256], [0, 256])
#     cr_hist = cv2.calcHist([ycrcb], [1], None, [256], [0, 256])
#     cb_hist = cv2.calcHist([ycrcb], [2], None, [256], [0, 256])

#     # RGB Image Plot
#     plt.subplot(4, 1, 1)
#     plt.imshow(rgb)
#     plt.title('RGB Image')
#     plt.xticks([])
#     plt.yticks([])

#     plt.subplot(4, 1, 2)
#     plt.plot(red_hist, color='r')
#     plt.xlim([0, 256])
#     plt.ylim([0, 500])
#     plt.title('Red Histogram')

#     plt.subplot(4, 1, 3)
#     plt.plot(green_hist, color='g')
#     plt.xlim([0, 256])
#     plt.ylim([0, 500])
#     plt.title('Green Histogram')

#     plt.subplot(4, 1, 4)
#     plt.plot(blue_hist, color='b')
#     plt.xlim([0, 256])
#     plt.ylim([0, 500])
#     plt.title('Blue Histogram')

#     plt.tight_layout()
#     #plt.show()

    r, g, b = cv2.split(rgb)
    
    r *= 255
    g *= 255
    b *= 255
    
    if save_color_images:
        RGB_PATH = os.path.join(FOLDER_PATH, FOLDER_NAME + '_RGB_' + version)
        if not os.path.exists(RGB_PATH):
            os.makedirs(RGB_PATH)
        
        cv2.imwrite(os.path.join(RGB_PATH, os.path.splitext(os.path.basename(path))[0]) + '_R_RGB_' + version + '_' + str(face_index) + '.png', r)
        cv2.imwrite(os.path.join(RGB_PATH, os.path.splitext(os.path.basename(path))[0]) + '_G_RGB_' + version + '_' + str(face_index) + '.png', g)
        cv2.imwrite(os.path.join(RGB_PATH, os.path.splitext(os.path.basename(path))[0]) + '_B_RGB_' + version + '_' + str(face_index) + '.png', b)
    
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

#     # HSV Image Plot
#     plt.subplot(4, 1, 1)
#     #plt.imshow(hsv)
#     plt.title('HSV Image')
#     plt.xticks([])
#     plt.yticks([])

#     plt.subplot(4, 1, 2)
#     plt.plot(hue_hist_HSV, color='c')
#     plt.xlim([0, 256])
#     plt.ylim([0, 2500])
#     plt.title('Hue Histogram')

#     plt.subplot(4, 1, 3)
#     plt.plot(saturation_hist_HSV, color='m')
#     plt.xlim([0, 256])
#     plt.ylim([0, 1000])
#     plt.title('Saturation Histogram')

#     plt.subplot(4, 1, 4)
#     plt.plot(value_hist, color='y')
#     plt.xlim([0, 256])
#     plt.ylim([0, 1000])
#     plt.title('Value Histogram')

#     plt.tight_layout()
#     plt.show()
    
    h, s, v = cv2.split(hsv)
    
    s *= 255
    v *= 255
    
    if save_color_images:
        HSV_PATH = os.path.join(FOLDER_PATH, FOLDER_NAME + '_HSV_' + version)
        if not os.path.exists(HSV_PATH):
            os.makedirs(HSV_PATH)
        
        cv2.imwrite(os.path.join(HSV_PATH, os.path.splitext(os.path.basename(path))[0]) + '_H_HSV_' + version + '_' + str(face_index) + '.png', h)
        cv2.imwrite(os.path.join(HSV_PATH, os.path.splitext(os.path.basename(path))[0]) + '_S_HSV_' + version + '_' + str(face_index) + '.png', s)
        cv2.imwrite(os.path.join(HSV_PATH, os.path.splitext(os.path.basename(path))[0]) + '_V_HSV_' + version + '_' + str(face_index) + '.png', v)
    
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
    
#     # HLS Image Plot
#     plt.subplot(4, 1, 1)
#     plt.imshow(hls)
#     plt.title('HLS Image')
#     plt.xticks([])
#     plt.yticks([])

#     plt.subplot(4, 1, 2)
#     plt.plot(hue_hist_HLS, color='r')
#     plt.xlim([0, 256])
#     plt.ylim([0, 2500])
#     plt.title('Hue Histogram')

#     plt.subplot(4, 1, 3)
#     plt.plot(lightness_hist_HLS, color='g')
#     plt.xlim([0, 256])
#     plt.ylim([0, 1000])
#     plt.title('Lightness Histogram')

#     plt.subplot(4, 1, 4)
#     plt.plot(saturation_hist_HLS, color='b')
#     plt.xlim([0, 256])
#     plt.ylim([0, 1000])
#     plt.title('Saturation Histogram')

#     plt.tight_layout()
#     plt.show()

    h, l, s = cv2.split(hls)
    
    l *= 255
    s *= 255
    
    if save_color_images:
        HSL_PATH = os.path.join(FOLDER_PATH, FOLDER_NAME + '_HSL_' + version)
        if not os.path.exists(HSL_PATH):
            os.makedirs(HSL_PATH)
        
        cv2.imwrite(os.path.join(HSL_PATH, os.path.splitext(os.path.basename(path))[0]) + '_H_HSL_' + version + '_' + str(face_index) + '.png', h)
        cv2.imwrite(os.path.join(HSL_PATH, os.path.splitext(os.path.basename(path))[0]) + '_S_HSL_' + version + '_' + str(face_index) + '.png', s)
        cv2.imwrite(os.path.join(HSL_PATH, os.path.splitext(os.path.basename(path))[0]) + '_L_HSL_' + version + '_' + str(face_index) + '.png', l)
    
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
    
#     # LAB Image Plot
#     plt.subplot(4, 1, 1)
#     plt.imshow(lab)
#     plt.title('LAB Image')
#     plt.xticks([])
#     plt.yticks([])

#     plt.subplot(4, 1, 2)
#     plt.plot(lightness_hist_LAB, color='c')
#     plt.xlim([0, 256])
#     plt.ylim([0, 1000])
#     plt.title('Lightness Histogram')

#     plt.subplot(4, 1, 3)
#     plt.plot(a_hist_LAB, color='m')
#     plt.xlim([0, 256])
#     plt.ylim([0, 20000])
#     plt.title('A Histogram')

#     plt.subplot(4, 1, 4)
#     plt.plot(b_hist_LAB, color='y')
#     plt.xlim([0, 256])
#     plt.ylim([0, 20000])
#     plt.title('B Histogram')

#     plt.tight_layout()
#     plt.show()
    
    l, a, b = cv2.split(lab)
    
    if save_color_images:
        LAB_PATH = os.path.join(FOLDER_PATH, FOLDER_NAME + '_LAB_' + version)
        if not os.path.exists(LAB_PATH):
            os.makedirs(LAB_PATH)
        
        cv2.imwrite(os.path.join(LAB_PATH, os.path.splitext(os.path.basename(path))[0]) + '_L_LAB_' + version + '_' + str(face_index) + '.png', l)
        cv2.imwrite(os.path.join(LAB_PATH, os.path.splitext(os.path.basename(path))[0]) + '_A_LAB_' + version + '_' + str(face_index) + '.png', a)
        cv2.imwrite(os.path.join(LAB_PATH, os.path.splitext(os.path.basename(path))[0]) + '_B_LAB_' + version + '_' + str(face_index) + '.png', b)
    
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
    
#     # YCrCb Image Plot
#     plt.subplot(4, 1, 1)
#     plt.imshow(ycrcb)
#     plt.title('YCrCb Image')
#     plt.xticks([])
#     plt.yticks([])

#     plt.subplot(4, 1, 2)
#     plt.plot(y_hist, color='r')
#     plt.xlim([0, 256])
#     plt.ylim([0, 1000])
#     plt.title('Y Histogram')

#     plt.subplot(4, 1, 3)
#     plt.plot(cr_hist, color='g')
#     plt.xlim([0, 256])
#     plt.ylim([0, 20000])
#     plt.title('Cr Histogram')

#     plt.subplot(4, 1, 4)
#     plt.plot(cb_hist, color='b')
#     plt.xlim([0, 256])
#     plt.ylim([0, 20000])
#     plt.title('Cb Histogram')

#     plt.tight_layout()
#     plt.show()
    
    y, cr, cb = cv2.split(ycrcb)
    
    y *= 255
    cr *= 255
    cb *= 255
    
    if save_color_images:
        YCRCB_PATH = os.path.join(FOLDER_PATH, FOLDER_NAME + '_YCRCB_' + version)
        if not os.path.exists(YCRCB_PATH):
            os.makedirs(YCRCB_PATH)
        
        cv2.imwrite(os.path.join(YCRCB_PATH, os.path.splitext(os.path.basename(path))[0]) + '_Y_YCRCB_' + version + '_' + str(face_index) + '.png', y)
        cv2.imwrite(os.path.join(YCRCB_PATH, os.path.splitext(os.path.basename(path))[0]) + '_CR_YCRCB_' + version + '_' + str(face_index) + '.png', cr)
        cv2.imwrite(os.path.join(YCRCB_PATH, os.path.splitext(os.path.basename(path))[0]) + '_CB_YCRCB_' + version + '_' + str(face_index) + '.png', cb)
    
    
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
    # rows = itertools.zip_longest([path], [face_index], r_hist, g_hist, b_hist, h_hist_HSV, s_hist_HSV, v_hist_HSV, h_hist_HSL, s_hist_HSL, l_hist_HSL, l_hist_LAB, a_hist_LAB, b_hist_LAB, y_hist, cr_hist, cb_hist)
    
    # with open("color.csv", "a", newline = "") as f:
    #     if os.stat("color.csv").st_size == 0:
    #         csv.writer(f).writerow(["Path", "Face Index", "Red", "Green", "Blue", "Hue_HSV", "Saturation_HSV", "Value_HSV", "Hue_HSL", "Saturation_HSL", "Lightness_HSL", "Lightness_LAB", "A_LAB", "B_LAB", "Y", "Cr", "Cb"])
    #     csv.writer(f).writerows(rows)
    
    return r_hist, g_hist, b_hist, h_hist_HSV, s_hist_HSV, v_hist_HSV, h_hist_HSL, s_hist_HSL, l_hist_HSL, l_hist_LAB, a_hist_LAB, b_hist_LAB, y_hist, cr_hist, cb_hist

# From https://medium.com/mlearning-ai/color-shape-and-texture-feature-extraction-using-opencv-cb1feb2dbd73
# Extracts Local Binary Pattern (Texture) of an image
def extract_lbp(path, face_index, image, version):
    # reads the input image as a grayscale image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    desc = LocalBinaryPatterns(24, 8)
    lbp_hist, lbp_img = desc.describe(gray)

    if save_lbp_images:
        LBP_PATH = os.path.join(FOLDER_PATH, FOLDER_NAME + '_LBP_' + version)
        if not os.path.exists(LBP_PATH):
            os.makedirs(LBP_PATH)
        
        cv2.imwrite(os.path.join(LBP_PATH, os.path.splitext(os.path.basename(path))[0]) + '_LBP_' + version + '_' + str(face_index) + '.png', lbp_img)
    
    # plt.imshow(lbp_img, cmap = plt.get_cmap('gray'))
    # plt.show()
    # lbp_hist = cv2.calcHist([lbp_img], [0], None, [256], [0, 256])
    
    # face_index = str(face_index)
    # rows = itertools.zip_longest([path], [face_index], lbp_hist)
    
    # with open("lbp.csv", "a", newline = "") as f:
    #     if os.stat("lbp.csv").st_size == 0:
    #         csv.writer(f).writerow(["Path", "Face Index", "LBP"])
    #     csv.writer(f).writerows(rows)
    
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
    
    if save_gradient_images:
        SOBEL_PATH = os.path.join(FOLDER_PATH, FOLDER_NAME + '_SOBEL_' + version)
        if not os.path.exists(SOBEL_PATH):
            os.makedirs(SOBEL_PATH)
        
        cv2.imwrite(os.path.join(SOBEL_PATH, os.path.splitext(os.path.basename(path))[0]) + '_SOBELX_' + version + '_' + str(face_index) + '.png', sobelx)
        cv2.imwrite(os.path.join(SOBEL_PATH, os.path.splitext(os.path.basename(path))[0]) + '_SOBELY_' + version + '_' + str(face_index) + '.png', sobely)
        cv2.imwrite(os.path.join(SOBEL_PATH, os.path.splitext(os.path.basename(path))[0]) + '_SOBEL_' + version + '_' + str(face_index) + '.png', sobel)

    # # display sobelx, sobely, and sobel
    # plt.imshow(sobelx, cmap = "gray")
    # plt.show()
    # plt.imshow(sobely, cmap = "gray")
    # plt.show()
    # plt.imshow(sobel, cmap = "gray")
    # plt.show()
    
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
    
    # face_index = str(face_index)
    # rows = itertools.zip_longest([path], [face_index], sobelx_hist, sobely_hist, sobel_hist)
    
    # with open("gradient.csv", "a", newline = "") as f:
    #     if os.stat("gradient.csv").st_size == 0:
    #         csv.writer(f).writerow(["Path", "Face Index", "Sobel X", "Sobel Y", "Sobel"])
    #     csv.writer(f).writerows(rows)
    
    return sobelx_hist, sobely_hist, sobel_hist

def extract_image_attributes(path, face_index, image, version):
    row = {}
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
    
def get_features(path, face_segmentor=None, extract_from_mask=False, fgsm_loss_target="conf", model=None, verbose=False):
    if model is None:
        main_yf = YoloFace()
        device, model = main_yf.device, main_yf.yf_face_detector

    torch.autograd.set_detect_anomaly(True)
    
    df = pd.DataFrame() # dataframe storing the dataset
    row = {} #the information/columns for a single row in the dataset is stored here
    grads = []
    bboxes = []
    masks = []
    
    file_basename = os.path.basename(path)
    if verbose:
        print(file_basename, end=" ")
        print("<- working on")

    row['path'] = path

    model.eval()
    model.gradient_mode = False

    for yolo_layer in model.yolo_layers:
        yolo_layer.gradient_mode = False

    # Read and transform the image from the path
    data = cv2.imread(path)
    row['source_w'], row['source_h'], _ = data.shape
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    data = transforms.Compose([DEFAULT_TRANSFORMS,Resize(416)])((data, np.zeros((1, 5))))[0].unsqueeze(0)

    with torch.no_grad():
        # Forward pass the data through the model and call non max suppression
        nms, nms_output = non_max_suppression(model(data), 0.5, 0.5) #conf_thres and iou_thres = 0.5

    face_list = []
    if type(nms_output[0]) is not int:
        face_list = nms_output[0]

    data = data.to(device)

    # Set requires_grad attribute of tensor. Important for attack
    data.requires_grad = True

    model.gradient_mode = True
    for yolo_layer in model.yolo_layers:
        yolo_layer.gradient_mode = True

    output = model(data)

    # loop through each of the faces in the image
    for face_index, face_row in enumerate(face_list): #nms_output[0] because the model is designed to take in several images at a time from the dataloader but we are only loading the image one at a time

        row['face_index'] = face_index
        if verbose:
            print("Face", face_index)

        row['obj_score'] = face_row[4].item()
        row['class_score'] = face_row[5].item()
        x, y, w, h = face_row[0], face_row[1], face_row[2], face_row[3]

        normal_x, normal_y, normal_w, normal_h = x / 415, y / 415, w / 415, h / 415

        if fgsm_loss_target == "bbox":
            target = torch.tensor([[face_row[4].item(), face_row[5].item(), 0, 0, 0, 0]])
        elif fgsm_loss_target == "conf":
            target = torch.tensor([[0.0, 0, normal_x, normal_y, normal_w, normal_h]])

        target = target.to(device)
        loss, loss_components = compute_loss(output, target, model)

        # cropped image with bounding box
        # getting (x1, y1) upper left, (x2, y2) lower right
        x1 = max(int(np.floor((x - w / 2).detach().cpu().numpy())), 0)
        y1 = max(int(np.floor((y - h / 2).detach().cpu().numpy())), 0)
        x2 = min(int(np.ceil((x + w / 2).detach().cpu().numpy())), 415)
        y2 = min(int(np.ceil((y + h / 2).detach().cpu().numpy())), 415)

        row['x1'], row['y1'], row['x2'], row['y2'] = x1, y1, x2, y2
        row['x'], row['y'], row['w'], row['h'] = x, y, w, h
        
        cropped_image = detach_cpu(data)[:, :, y1:y2, x1:x2] #get the first dimension, the channels, and crop it
        cropped_image = tensor_to_np_img(cropped_image) #reshape the image to (w/h, h/w, channel)

        # Zero all existing gradients
        model.zero_grad()
        data.grad = None

        # Calculate gradients of model in backward pass
        loss.backward(retain_graph=True) #TODO: Amos - check if this is correct
        
        # Collect datagrad
        data_grad = data.grad.data
        grads.append(clone_detach(data_grad))
        
        bbox = (x1, y1, x2, y2)
        bboxes.append(bbox)
        
        if face_segmentor is None:
            mask = np.ones((cropped_image.shape[0], cropped_image.shape[1]))
        else:
            mask, used_mask = face_segmentor.segment_transform(cropped_image)
        
        if not extract_from_mask:
            row = {**row, **extract_image_attributes(path, face_index, cropped_image, "bbox")}
        else:
            masked_image = cropped_image
            masked_image[mask == 0] = 0
            row = {**row, **extract_image_attributes(path, face_index, masked_image, "mask")}
        
        whole_mask = np.zeros(data.shape)
        whole_mask[..., bbox[1]:bbox[3], bbox[0]:bbox[2]] = mask
        
        masks.append(torch.Tensor(whole_mask))
        
        df = pd.concat([df, pd.DataFrame([row])], axis=0, ignore_index=True)
    
    return df, grads, bboxes, masks