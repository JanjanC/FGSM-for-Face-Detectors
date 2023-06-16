import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import time
import glob
import gdown
import torch
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import shutil
from PIL import Image

#libraries for yolo
from pytorchyolo.models import load_model
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
from pytorchyolo.utils.utils import non_max_suppression

def detach_cpu(image):
    return image.detach().cpu()

# convert 1x3x416x416 to 416x416x3
def reshape_image(image):
    return np.transpose(np.squeeze(image), (1 ,2, 0))

# convert 1x3x416x416 tensor to 416x416x3 numpy image
def tensor_to_image(image):
    return np.transpose(image.detach().cpu().squeeze().numpy(), (1, 2, 0))

_, yolo_model = load_model('./weights/yolo_face_sthanhng.cfg', "./weights/yolo_face_sthanhng.weights")
# Set the model in evaluation mode. In this case this is for the Dropout layers
yolo_model.eval()
def pipeline(img_path):
    
    df = pd.DataFrame() # dataframe storing the dataset
    row = {} #the information/columns for a single row in the dataset is stored here
    
        
    # GET THE FILENAME
    filename = os.path.basename(img_path)
    filename = filename.split(".")[0]
    print(filename)

    # INITIALIZE A FACE COUNTER
    face_count = 0

    row['path'] = img_path
    row['source_file'] = img_path.split("\\")[-1]
    # print(path)
    # print(row['source_file'])

    # read and transform the image from the path
    data = cv2.imread(img_path)  # read the image
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB) #change to rgb
    data = transforms.Compose([DEFAULT_TRANSFORMS,Resize(416)])((data, np.zeros((1, 5))))[0].unsqueeze(0) # transform the image

    with torch.no_grad():
    # Forward pass the data through the model
        output = yolo_model(data)

        # call non max suppression
        nms, nms_output = non_max_suppression(output, 0.5, 0.5) #conf_thres and iou_thres = 0.5

    face_list = []
    if type(nms_output[0]) is not int:
        face_list = nms_output[0]

    # loop through each of the faces in the image
    for face_index, face_row in enumerate(face_list): #nms_output[0] because the model is designed to take in several images at a time from the dataloader but we are only loading the image one at a time
        # FACE_FILENAME
        face_filename = filename + "_" + str(face_count)
        row['face_index'] = face_index
#             print('Face ', face_index)

        # get the coordinate of the face bounding box
        #(x1, y1) upper left, (x2, y2) lower right
        x, y, w, h = face_row[0], face_row[1], face_row[2], face_row[3]

        # cropped image with bounding box
        # getting (x1, y1) upper left, (x2, y2) lower right
        x1 = max(int(np.floor((x - w / 2).detach().cpu().numpy())), 0)
        y1 = max(int(np.floor((y - h / 2).detach().cpu().numpy())), 0)
        x2 = min(int(np.ceil((x + w / 2).detach().cpu().numpy())), 415)
        y2 = min(int(np.ceil((y + h / 2).detach().cpu().numpy())), 415)

        row['x1'], row['y1'], row['x2'], row['y2'] = x1, y1, x2, y2

        # print('Cropped')
        # print(x1, y1, x2, y2)
        bbox_image = detach_cpu(data)[:, :, y1:y2, x1:x2] #get the first dimension, the channels, and crop it
        bbox_image = tensor_to_image(bbox_image) #reshape the image to (w/h, h/w, channel)
        bbox_image = np.transpose(transforms.Compose([DEFAULT_TRANSFORMS,Resize(128)])((bbox_image, np.zeros((1, 5))))[0], (1, 2, 0)).numpy() # resize image to 128x128
        bbox_image = (bbox_image * 255).astype(np.uint8)
#             plt.imshow(bbox_image)
#             plt.show()

#             bbox_image = cv2.cvtColor(bbox_image, cv2.COLOR_RGB2BGR)
#             cv2.imwrite(os.path.join(BBOX_PATH, row['source_file'] + str(face_index) + '.jpg'), bbox_image)

        # getting (x1, y1) upper left, (x2, y2) lower right
         # cropped image with bounding box + some padding
        x1_pad = max(int(np.floor((x - w).detach().cpu().numpy())), 0) # prevent negative values
        y1_pad = max(int(np.floor((y - h).detach().cpu().numpy())), 0)
        x2_pad = min(int(np.ceil((x + w).detach().cpu().numpy())), 415) # prevent from getting out of range
        y2_pad = min(int(np.ceil((y + h).detach().cpu().numpy())), 415)

        row['x1_pad'], row['y1_pad'], row['x2_pad'], row['y2_pad'] = x1_pad, y1_pad, x2_pad, y2_pad

        pad_image = detach_cpu(data)[:, :, y1_pad:y2_pad, x1_pad:x2_pad] #get the first dimension, the channels, and crop it
        pad_image = tensor_to_image(pad_image) #reshape the image to (w/h, h/w, channel)

        # Original Pad Size
        # GET THE LONGEST MAX AND THEN PAD
        greater_size = max((x2_pad - x1_pad),(y2_pad - y1_pad))
        orig_pad_image = np.transpose(transforms.Compose([DEFAULT_TRANSFORMS,Resize(greater_size)])((pad_image, np.zeros((1, 5))))[0], (1, 2, 0)).numpy() # resize image to GREATER SIZE
        orig_pad_image = (orig_pad_image * 255).astype(np.uint8)
        orig_pad_image = cv2.cvtColor(orig_pad_image, cv2.COLOR_RGB2BGR)         
        cv2.imwrite(os.path.join(ORIG_PAD_PATH, "mask_" + face_filename + "_image_final.png"), orig_pad_image)

        pad_image = np.transpose(transforms.Compose([DEFAULT_TRANSFORMS,Resize(128)])((pad_image, np.zeros((1, 5))))[0], (1, 2, 0)).numpy() # resize image to 128x128
        pad_image = (pad_image * 255).astype(np.uint8)
#             plt.imshow(pad_image)
#             plt.show()

        # SAVE THE ORIGINAL NAME OF THE IMAGE FIRST.... AND THEN FOR EACH FACE APPEND THE FACE COUNTER            
        face_count += 1

        # ADD TO ROW FACE_FILENAME
        row['mask_filename'] = "mask_" + face_filename + "_image_final.png"

        # SAVE AS 24-BIT PNG WITH THE FORMAT OF IMAGEFILENAME_NO STUFF AND SUFFIX
        im = Image.fromarray(pad_image)
        im.save(PROCESS_PATH + "\\" +  face_filename + "_image_final.png")

        # [DUPLICATE WITH THE BLACK.PNG]            
        # Create the black _cc_occ_labels and _sp_labels (16 bit pngs)
        cc_occ_png = PROCESS_PATH + "\\" +  face_filename + "_cc_occ_labels.png"
        sp_png = PROCESS_PATH + "\\" +  face_filename + "_sp_labels.png"
        mask_png = PROCESS_PATH + '\\' + face_filename + "_mask.png"

        # black.png is reference image being duplicated
        shutil.copyfile("black.png", (cc_occ_png))
        shutil.copyfile("black.png", (sp_png))    

        pad_image = cv2.cvtColor(pad_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(PAD_PATH, row['source_file'] + str(face_index) + '.jpg'), pad_image)

        #add tow to dataset
        df = df.append(row, ignore_index=True) #append the attributes of one face to the dataframe                              
    df.to_csv(os.path.join(CSV_PATH, 'dataset' + '.csv'), index=False)  #save to csv
    
import sys
import os
import torch
import torchvision
import numpy as np
import pandas as pd
sys.path.append('src_release')

#libraries for face segmentation
from data_loader import get_dataloader
from models.encoder_decoder_faceoccnet import FaceOccNet 
from torch_utils import torch_load_weights,evaluation,viz_notebook,plot_confusion_matrix

load_model_path = ("./ptlabel_best_model.pth")
fs_model = FaceOccNet(input_channels=3, n_classes=3,is_regularized=True)

fs_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fs_model.to(fs_device)

# You have to set def load in serialization.py to have its map_location parameter = 'cpu'
if os.path.exists(load_model_path) and os.path.isfile(load_model_path):
    _, _ = torch_load_weights(fs_model, None, load_model_path, model_only=True)
    print(f'Loaded model from {load_model_path}')
else:
    print(f'The model does not exist in {load_model_path} or is not a file')
    
from tqdm import tqdm as fs_tqdm
from skimage import measure
import matplotlib.pyplot as plt
from data_tools import decode_mask2img,encode_img2mask

'''
Visualization function for tensorboard and notebook
'''
def tf_viz_img(mask_tmp,i,pred=True):
    if pred:
        mask_tmp = torch.argmax(mask_tmp[i], dim=0).numpy().copy()
    else:
        mask_tmp = mask_tmp[i].numpy().copy()
    mask_tmp = decode_mask2img(mask_tmp)
    mask_tmp = np.transpose(mask_tmp, (2,0,1))
    mask_tmp = mask_tmp / 255.0
    return mask_tmp

'''
Main Visualization function
'''
def viz_notebook_brew(fs_model,eval_dataloader,fs_device,ibv_stop=-1):
    import matplotlib.pyplot as plt
    unorm = torchvision.transforms.Compose([ torchvision.transforms.Normalize((-1, -1, -1), (2, 2, 2))])
    #batch_val = iter(eval_dataloader).next()
    fs_model.eval()
    
    with torch.no_grad():
        for ibv, batch_val in fs_tqdm(enumerate(eval_dataloader),
                               desc='viz'):
            fs_img, mask_gt, mask_sp, fn = batch_val
            pred_mask, _ = fs_model(fs_img.to(fs_device))
            pred_mask = pred_mask.cpu()
            mask_gt = mask_gt.cpu().data
            face_count = 0
            for b in range(pred_mask.shape[0]):
                pred_tmp = tf_viz_img(pred_mask,b,pred=True)
                # mask_gt_tmp = tf_viz_img(mask_gt,b,pred=False)
                pred_tmp = np.transpose(pred_tmp, (1,2,0))
                # mask_gt_tmp = np.transpose(mask_gt_tmp, (1,2,0))
                ##plotting
#                 fig = plt.figure()
#                 plt.subplot(1,3,1)
#                 plt.title(f'Image {fs_img[b].shape[2]}')
#                 plt.imshow(np.transpose(unorm(fs_img[b]), (1,2,0)))
#                 plt.axis('off')
                print("IMAGE FILENAME IS: " + fn[face_count])
                                
                # TURN THE BLUE AND GREEN PRED_TMP TO WHITE
                # Convert non-black pixels to white
                non_black_pixels_mask = np.any(pred_tmp != [0, 0, 0], axis=-1)  
                pred_tmp[non_black_pixels_mask] = [1, 1, 1]     
                
#                 plt.subplot(1,3,2)
#                 plt.title(f'Prediction {pred_tmp.shape[0]}')
#                 plt.imshow(pred_tmp)
#                 plt.axis('off')                                                                                
                plt.imsave(MASK_PATH + '\\' + 'mask_' + fn[face_count], pred_tmp)
                
                face_count+=1
                
#                 plt.show()
#                 plt.close(fig)
            if ibv_stop == ibv:
                break       

OUTPUT_FOLDER = os.path.join(os.getcwd(), "_temp")
if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)

BBOX_PATH = os.path.join(os.getcwd(), OUTPUT_FOLDER, '_box')
PAD_PATH = os.path.join(os.getcwd(), OUTPUT_FOLDER, '_pad')
ORIG_PAD_PATH = os.path.join(os.getcwd(), OUTPUT_FOLDER, '_original_pad')
PROCESS_PATH = os.path.join(os.getcwd(), OUTPUT_FOLDER, '_process')
CSV_PATH = os.path.join(os.getcwd(), OUTPUT_FOLDER, '_csv')
MASK_PATH = os.path.join(os.getcwd(), OUTPUT_FOLDER, '_mask')

def create_folders():
    if not os.path.exists(BBOX_PATH):
        os.mkdir(BBOX_PATH)

    if not os.path.exists(BBOX_PATH):
        os.mkdir(BBOX_PATH)

    if not os.path.exists(PAD_PATH):
        os.mkdir(PAD_PATH)

    if not os.path.exists(ORIG_PAD_PATH):
        os.mkdir(ORIG_PAD_PATH)

    if not os.path.exists(PROCESS_PATH):
        os.mkdir(PROCESS_PATH)

    if not os.path.exists(CSV_PATH):
        os.mkdir(CSV_PATH)

    if not os.path.exists(MASK_PATH):
        os.mkdir(MASK_PATH)
    
def create_mask(img_path):
    create_folders()
    pipeline(img_path)
    counter = 0
    
    while len(glob.glob(os.path.join(PROCESS_PATH, '*.png'))) > 0:
        counter = counter + 1

        #create intermediate folder
        INTERMEDIATE_PATH = os.path.join(os.getcwd(),OUTPUT_FOLDER, '_intermediate_' + str(counter))
        if not os.path.exists(INTERMEDIATE_PATH):
            os.mkdir(INTERMEDIATE_PATH)

        # loop through the images in process path
        for index, path in enumerate(glob.glob(os.path.join(PROCESS_PATH, '*.png'))):

            if index >= 200 * 3:
                break

            file = os.path.basename(path) #extract file name
            source = os.path.join(PROCESS_PATH, file) # source + file name
            destination =  os.path.join(INTERMEDIATE_PATH, file) # destination + file name

            os.rename(source, destination) #move images to the destination folder

        eval_dataloader = get_dataloader((INTERMEDIATE_PATH,),
                              batch_size=250,
                              mode='eval', 
                              num_workers = 4,
                              n_classes=3,
                              dataset_name='PartLabel')

        viz_notebook_brew(fs_model,eval_dataloader,fs_device,ibv_stop=0)

    # -----

    RESTORED_MASK_PATH = os.path.join(os.getcwd(), OUTPUT_FOLDER, '_restored_mask')
    APPLIED_MASK_PATH = os.path.join(os.getcwd(), OUTPUT_FOLDER, '_applied_mask')
    if not os.path.exists(RESTORED_MASK_PATH):
        os.mkdir(RESTORED_MASK_PATH)

    if not os.path.exists(APPLIED_MASK_PATH):
        os.mkdir(APPLIED_MASK_PATH)    

    # LOAD DATASET
    faces_df = pd.read_csv(CSV_PATH + "\\dataset.csv")
    pixels_df = pd.DataFrame()
    no_whites_df = pd.DataFrame()

    # PER MASK FILES IN LOCATION
    masks = glob.glob(MASK_PATH + "/*.png")
    for mask in masks:
        with open(mask, 'rb') as file:
            # GET THE FILENAME       
            filename = os.path.basename(MASK_PATH + mask)         
            print(filename)

            # MATCH FILENAME WITH DATASET
            face_row = faces_df.loc[faces_df['mask_filename'] == filename]        

            # SET THE RESTORED FILENAME 
            restored_filename = "restored_" + filename                
            face_row['filename'] = restored_filename  

            # GET THE ORIGINAL SIZE
            x_size = face_row['x2_pad'] - face_row['x1_pad']
            y_size = face_row['y2_pad'] - face_row['y1_pad']        
            greater_size = max(x_size.item(), y_size.item())        
            restored_size = (greater_size, greater_size)                

            # OPEN AND PROCESS IMAGE
            img = Image.open(file).convert("RGB")        
            img = img.resize(restored_size)

            img_arr = np.array(img)

            # Convert non-black pixels to white
            non_black_pixels_mask = np.any(img_arr != [0, 0, 0], axis=-1)          
            img_arr[non_black_pixels_mask] = [255, 255, 255]

            # CHECK NUMBER OF PIXELS AND UNIQUE VALUES WITH THIS
            # unique, counts = np.unique(img_arr, return_counts=True)
            # print(np.asarray((unique, counts)).T)
            # CHECK UNIQUE VALUES WITH THIS
            # with np.printoptions(threshold=np.inf):
            #     print(img_arr)

            # COUNT PIXELS FASTER ALTERNATIVE
            unique, counts = np.unique(img_arr, return_counts=True)
            if unique.size == 2:
                total_pixels = np.asarray((unique, counts)).T[1][1] / 3
                face_row['pixels'] = total_pixels
            else:
                total_pixels = 0
                face_row['pixels'] = total_pixels
                no_whites_df = no_whites_df.append(face_row, ignore_index=True)            
                # Convert everything to white
                black_pixels_mask = np.any(img_arr == [0, 0, 0], axis=-1)
                img_arr[black_pixels_mask] = [255, 255, 255]            

            pixels_df = pixels_df.append(face_row, ignore_index=True)

            # SAVE THE FILE
            sum_img = Image.fromarray(img_arr)
            sum_img = sum_img.save(RESTORED_MASK_PATH + "\\" + restored_filename)

            # MASKING HERE   
            # OPEN THE FILE AND MATCH THE FILENAME 
            img_orig = Image.open(ORIG_PAD_PATH + "\\" + filename)
            img_orig = np.array(img_orig)        
            # Select the location of all black pixels
            black_pixels_mask = np.any(img_arr == [0, 0, 0], axis=-1)
            img_orig[black_pixels_mask] = [0, 0, 0]     

            sum_img_orig = Image.fromarray(img_orig)
            sum_img_orig = sum_img_orig.save(APPLIED_MASK_PATH + "\\" + filename)

    pixels_df.to_csv(os.path.join(CSV_PATH, 'dataset_pixels' + str(int(time.time())) + '.csv'), index=False)  #save to csv        
    no_whites_df.to_csv(os.path.join(CSV_PATH, 'dataset_no_whites' + str(int(time.time())) + '.csv'), index=False)  #save to csv 
    
    purge_folders()
    
    
def purge_folders():
    OUTPUT_FOLDER = os.path.join(os.getcwd(), "_temp")
    
    
    
    BBOX_PATH = os.path.join(os.getcwd(), OUTPUT_FOLDER, '_box')
    if os.path.exists(BBOX_PATH):
        shutil.rmtree(BBOX_PATH)
        
    PAD_PATH = os.path.join(os.getcwd(), OUTPUT_FOLDER, '_pad')
    if os.path.exists(PAD_PATH):
        shutil.rmtree(PAD_PATH)
        
    ORIG_PAD_PATH = os.path.join(os.getcwd(), OUTPUT_FOLDER, '_original_pad')
    if os.path.exists(ORIG_PAD_PATH):
        shutil.rmtree(ORIG_PAD_PATH)
        
    PROCESS_PATH = os.path.join(os.getcwd(), OUTPUT_FOLDER, '_process')
    if os.path.exists(PROCESS_PATH):
        shutil.rmtree(PROCESS_PATH)
        
    CSV_PATH = os.path.join(os.getcwd(), OUTPUT_FOLDER, '_csv')
    if os.path.exists(CSV_PATH):
        shutil.rmtree(CSV_PATH)
        
    MASK_PATH = os.path.join(os.getcwd(), OUTPUT_FOLDER, '_mask')
    if os.path.exists(MASK_PATH):
        shutil.rmtree(MASK_PATH)

    INTERMEDIATE_PATH = os.path.join(os.getcwd(),OUTPUT_FOLDER, '_intermediate_' + str(1))
    if os.path.exists(INTERMEDIATE_PATH):
        shutil.rmtree(INTERMEDIATE_PATH)
        
    APPLIED_MASK_PATH = os.path.join(os.getcwd(), OUTPUT_FOLDER, '_applied_mask')
    if os.path.exists(APPLIED_MASK_PATH):
        shutil.rmtree(APPLIED_MASK_PATH)