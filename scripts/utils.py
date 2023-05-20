import numpy as np
import gdown
import os
import torchvision.transforms as transforms
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS

def open_img_as_tensor(filename):
    data = cv2.imread(filename)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    data = transforms.Compose([DEFAULT_TRANSFORMS,Resize(416)])((data, np.zeros((1, 5))))[0].unsqueeze(0)
    return data

def tensor_to_np(image):
    return np.moveaxis((image.detach().numpy() * 255).squeeze(), 0, -1).astype('uint8')

def save_tensor_img(tensor, filename):
    save_img = cv2.cvtColor(tensor_to_np(tensor), cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, tensor)

def download_weight(filename, link):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    
    if not os.path.isfile(filename):
        print("Downloading:", filename)
        gdown.download(link, filename, quiet=False)