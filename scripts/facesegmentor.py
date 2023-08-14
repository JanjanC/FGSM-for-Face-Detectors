import os
import numpy as np
import sys
import torch
import torchvision
import torch.nn.functional as F
import cv2
from scripts.utils import np_to_tesor_img

sys.path.append('facesegmentor')
from models.encoder_decoder_faceoccnet import FaceOccNet 
from torch_utils import torch_load_weights,evaluation,viz_notebook,plot_confusion_matrix
from data_tools import decode_mask2img,encode_img2mask

WEIGHTS_DIR = os.path.join(os.getcwd(), 'face_detector_weights')

class FaceSegementor():
    def __init__(self, weight_loc=os.path.join(WEIGHTS_DIR, "faceseg", "ptlabel_best_model.pth"), weight_url="https://drive.google.com/file/d/1cdojb7o0E7VPiK_xFCtCa9ZpFYK_a9lM/view?usp=drive_link"):
        utils.download_weight(weight_loc, weight_url)
        fs_model = FaceOccNet(input_channels=3, n_classes=3,is_regularized=True)
        fs_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        fs_model.to(fs_device)
        torch_load_weights(fs_model, None, weight_loc, model_only=True)
        self.fs_model = fs_model
    
    def segment(self, image):
        orig_shape = image.shape
        #frame = np.zeros((max(orig_shape), max(orig_shape), orig_shape[-1]))
        #frame[:orig_shape[0], :orig_shape[1], :] = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.Tensor(np.moveaxis(image, -1, 0)).unsqueeze(0)
        image = image.cpu   ()
        pred_mask_orig, _ = self.fs_model(image)
        pred_mask_orig.shape
        pred_mask = pred_mask_orig.detach()
        unorm = torchvision.transforms.Compose([ torchvision.transforms.Normalize((-1, -1, -1), (2, 2, 2))])
        final = np.zeros((pred_mask.shape[-2], pred_mask.shape[-1], 3))
        for b in range(pred_mask.shape[0]):
            pred_tmp = torch.argmax(pred_mask[b], dim=0).numpy().copy()
            pred_tmp = decode_mask2img(pred_tmp)
            pred_tmp = np.transpose(pred_tmp, (2,0,1))
            pred_tmp = pred_tmp / 255.0
            pred_tmp = np.transpose(pred_tmp, (1,2,0))
            non_black_pixels_mask = np.any(pred_tmp != [0, 0, 0], axis=-1)  
            pred_tmp[non_black_pixels_mask] = [1, 1, 1]
            final += pred_tmp
        final = np.clip(final, 0, 1)
        #"""
        padding = (
            int(np.floor(np.abs(orig_shape[0] - final.shape[0]) / 2)),
            int(final.shape[0] - np.ceil(np.abs(orig_shape[0] - final.shape[0]) / 2)),
            int(np.floor(np.abs(orig_shape[1] - final.shape[1]) / 2)),
            int(final.shape[1] - np.ceil(np.abs(orig_shape[1] - final.shape[1]) / 2)),
        )
        return final[padding[0]:padding[1], padding[2]:padding[3], 0].astype('uint8')
        #"""
        #return final[:orig_shape[0], :orig_shape[1], 0].astype('uint8')
        
    def segment_transform(self, image):
        mask = self.segment(image)
        num_white = dict(zip(*np.unique(mask, return_counts = True)))[1]
        if num_white < int(mask.shape[0] * mask.shape[1] * 0.1):
            return np.ones((mask.shape[0], mask.shape[1])), False
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(mask.shape[0] * 0.5), int(mask.shape[1] * 0.5)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        return mask, True