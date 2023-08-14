import cv2
import numpy as np
import gdown
import os
from torch import Tensor
import torchvision.transforms as transforms
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
import matplotlib.pyplot as plt

def open_img_as_tensor(filename):
    image = cv2.imread(filename)
    if image is None:
        raise Exception("File could not be opened")
    return np_to_tesor_img(image)
    
def np_to_tesor_img(image):
    if image.dtype is not np.uint8:
        image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return transforms.Compose([DEFAULT_TRANSFORMS,Resize(416)])((image, np.zeros((1, 5))))[0].unsqueeze(0)

def tensor_to_np_img(image):
    return np.moveaxis((image.detach().numpy() * 255).squeeze(), 0, -1).astype('uint8')

def save_tensor_img(tensor, filename):
    save_img = cv2.cvtColor(tensor_to_np_img(tensor), cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, save_img)

def download_weight(filename, link):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    
    if not os.path.isfile(filename):
        print("Downloading:", filename)
        gdown.download(link, filename, quiet=False)

def detach_cpu(image):
    return image.detach().cpu()
    
def clone_detach(image):
    return image.clone().detach()
    
def remap( x, oMin, oMax, nMin, nMax ):

    #range check
    if oMin == oMax:
        print("Warning: Zero input range")
        return None

    if nMin == nMax:
        print("Warning: Zero output range")
        return None

    #check reversed input range
    reverseInput = False
    oldMin = min( oMin, oMax )
    oldMax = max( oMin, oMax )
    if not oldMin == oMin:
        reverseInput = True

    #check reversed output range
    reverseOutput = False   
    newMin = min( nMin, nMax )
    newMax = max( nMin, nMax )
    if not newMin == nMin :
        reverseOutput = True

    portion = (x-oldMin)*(newMax-newMin)/(oldMax-oldMin)
    if reverseInput:
        portion = (oldMax-x)*(newMax-newMin)/(oldMax-oldMin)

    result = portion + newMin
    if reverseOutput:
        result = newMax - portion

    return result
    
def display_img(*imgs, cmap=None):
    for img in imgs:
        if isinstance(img, Tensor):
            img = tensor_to_np_img(img)
        plt.imshow(img, cmap=cmap)
        plt.show()
        
def isstr(o):
  try:
    basestring
  except NameError:
    basestring = (str, bytes)
  return isinstance(o, basestring)

def toiter(o):
  if not isstr(o):
    try:
      return iter(o)
    except TypeError:
      pass
  return iter([o])