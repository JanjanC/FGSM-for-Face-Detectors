import os
import torch.nn.functional as F
import pandas as pd
import numpy as np
import csv
import cv2
import torchvision.transforms as transforms
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS

OUTPUT_FOLDER = os.path.join(os.getcwd(), "faceseg-outs")
folders = []

# REPLACE folders with your own folders

for FOLDER_NAME in folders:
    FOLDER_PATH = os.path.join(OUTPUT_FOLDER, FOLDER_NAME)
    CSV_PATH = os.path.join(FOLDER_PATH, FOLDER_NAME + '_CSV')
    RESTORED_MASK_PATH = os.path.join(FOLDER_PATH, FOLDER_NAME + '_restored_mask')

    CSV_FILE = ""
    NEW_CSV = ""
    for file in os.listdir(CSV_PATH):
        if "dataset_pixels" in file and file.endswith(".csv"):
            NEW_CSV = os.path.join(os.getcwd(), CSV_PATH, "actual_pixels_" + file)
            CSV_FILE = os.path.join(os.getcwd(), CSV_PATH, file)

    faces_df = pd.read_csv(CSV_FILE)
    faces_df.loc[:, ["x1", "y1", "x2", "y2", "x1_pad", "y1_pad", "x2_pad", "y2_pad"]] = faces_df.loc[:, ["x1", "y1", "x2", "y2", "x1_pad", "y1_pad", "x2_pad", "y2_pad"]].clip(lower = 0)
    actual_pixels = []
    print("working on", FOLDER_NAME)
    for _, row in faces_df.iterrows():
        mask = cv2.imread(os.path.join(os.getcwd(), RESTORED_MASK_PATH, row["filename"]), 0)
        padded_dim = (int(row["x2_pad"] - row["x1_pad"]), int(row["y2_pad"] - row["y1_pad"]))
        target_bbox = (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
        target_dim = (int(target_bbox[2] - target_bbox[0]), int(target_bbox[3] - target_bbox[1]))

        if dict(zip(*np.unique(mask, return_counts = True)))[255] < int(target_dim[0] * target_dim[1] * 0.1):
            actual_pixels.append(target_dim[1] * target_dim[0])
            continue
            
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(mask.shape[0] * 0.5), int(mask.shape[1] * 0.5)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        mask = transforms.Compose([DEFAULT_TRANSFORMS])((mask, np.zeros((1, 5))))[0].unsqueeze(0)

        current_dim = max(mask.shape)
        diff_x, diff_y = abs(padded_dim[0] - current_dim) / 2, abs(padded_dim[1] - current_dim) / 2

        if diff_y != 0:
            mask = mask[..., int(np.floor(diff_y)):-int(np.ceil(diff_y)), :]
        if diff_x != 0:
            mask = mask[..., int(np.floor(diff_x)):-int(np.ceil(diff_x))]

        padding = [
            int(abs(row["x1"] - row["x1_pad"])),
            int(abs(row["y1"] - row["y1_pad"])),
            int(abs(row["x2"] - row["x2_pad"])),
            int(abs(row["y2"] - row["y2_pad"]))
        ]

        new_dim = padded_dim[0] - padding[0] - padding[2], padded_dim[1] - padding[1] - padding[3]
        diff_x, diff_y = (target_dim[0] - new_dim[0]) / 2, (target_dim[1] - new_dim[1]) / 2

        padding[0] -= int(np.floor(diff_x))
        padding[1] -= int(np.floor(diff_y))
        padding[2] -= int(np.ceil(diff_x))
        padding[3] -= int(np.ceil(diff_y))
        mask = F.pad(input=mask, pad=(-padding[0], -padding[2], -padding[1], -padding[3]), mode='constant', value=0)
        actual_pixels.append(int(mask[0, 0, ...].sum().item()))
    faces_df["actual_pixels"] = actual_pixels
    faces_df.to_csv(NEW_CSV)