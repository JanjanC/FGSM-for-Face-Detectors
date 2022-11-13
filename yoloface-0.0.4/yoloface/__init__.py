import numpy as np
from PIL import Image
import cv2
import time
import os
import gdown
from IPython.display import display

class face_analysis:
    def __init__(self):
        '''
        After runing this cell , the weights of that model are stored as a file in the ".yoloface" folder
        We can also download  weights and cfg file from Google Drive.
        # folder structure:
          folder name: .yoloface
                      sub file 1 :yolov3-tiny_face.weights
                      sub file 1 :yolov3_tiny_face.cfg
                      sub file 1 :face_detection.weights
                      sub file 1 : face_detection.cfg
        #After running this cell, the weights will be available in our working directory as a folder called .yoloface .
        '''

        model_file=[
                    'yolov3-tiny_face.weights',
                    'yolov3_tiny_face.cfg',
                    'face_detection.weights',
                    'face_detection.cfg'
                    ]
        gdrive_url=[
                  'https://drive.google.com/uc?id=1JYrRT4Xe-NTrxYGhcj_hy7NtL4D2c77v',
                  'https://drive.google.com/uc?id=1S4W6mpOQgVvjpPupbhW6rHFjl-OaUR-g',
                  'https://drive.google.com/uc?id=1EFlRsOA6oGLLBH2VMDOqHSfDI70CDXeW',
                  'https://drive.google.com/uc?id=1ZxUmO0B1435taazz8P0Pqjq8r8Rypeip'
                  ]
      #            
        cwd=os.getcwd() 
        if '.yoloface' in os.listdir(cwd) :
            for i in range(len(model_file)):
              if model_file[i] in os.listdir(os.path.join(cwd,'.yoloface')):
                
                print( model_file[i]+':: status : file already exists')
              else:
                gdown.download(gdrive_url[i],os.path.join(cwd,'.yoloface',model_file[i]),quiet=False)
        else:
            os.makedirs(os.path.join(cwd,'.yoloface'))
            time.sleep(1)
            for i in range(len(model_file)):
              gdown.download(gdrive_url[i],os.path.join(cwd,'.yoloface',model_file[i]),quiet=False)    
              time.sleep(1)
        time.sleep(1)
        # tiny
        self.tiny_weight_path=os.path.join(os.getcwd(),'.yoloface','yolov3-tiny_face.weights') #weight file path
        self.tiny_cfg_path=os.path.join(os.getcwd(),'.yoloface','yolov3_tiny_face.cfg')
        self.tiny_net = cv2.dnn.readNet(self.tiny_weight_path,self.tiny_cfg_path)
        # full model
        self.full_weight_path=os.path.join(os.getcwd(),'.yoloface','face_detection.weights') #weight file path
        self.full_cfg_path=os.path.join(os.getcwd(),'.yoloface','face_detection.cfg')
        self.full_net = cv2.dnn.readNet(self.full_weight_path,self.full_cfg_path)

     ###
    def face_detection(self,image_path=None,model='full',frame_status=False,frame_arr=None):
        '''
        # example 1
        img,box,confidence=face.face_detection('path/to/jpg or png/file.jpg',model='full')
        print(len(box))
        print(len(confidence))
        face.show_output(img,box)
        ==================================================================================
        # example 2
        img,box,conf=face.face_detection(image_path='a56.jpg',model='tiny')
        print(box)
        print(conf)
        face.show_output(img,box)
        ===================================================================================
        # example 3
        cap = cv2.VideoCapture(0)
        while True: 
            _, frame = cap.read()
            _,box,conf=face.face_detection(frame=frame,frame_status=True,model='tiny')
            output_frame=face.show_output(frame,box,frame_status=True)
            cv2.imshow('frame',output_frame)
            key=cv2.waitKey(1)
            if key ==ord('v'): 
                break 
        cap.release()
        cv2.destroyAllWindows()
        #press v (exits)
        ==================================================================================
        # example 4
        cap = cv2.VideoCapture(0)
        #cap = cv2.VideoCapture(r'path/to/video/file/filname.mp4')
        while True: 
            _, frame = cap.read()
            _,box,conf=face.face_detection(frame_arr=frame,frame_status=True,model='tiny')
            output_frame=face.show_output(img=frame,face_box=box,frame_status=True)
            cv2.imshow('frame',output_frame)
            print(box)
            key=cv2.waitKey(1)
            if key ==ord('v'): 
                break 
        cap.release()
        cv2.destroyAllWindows()
        
         '''
        self.image_path=image_path
        self.frame_status=frame_status
        self.frame_arr=frame_arr
        if self.image_path!=None and self.frame_status==False and self.frame_arr==None:         
            self.image_path=image_path
            img=cv2.imread(self.image_path)
        else:
            img=self.frame_arr    
        
        height,width, _ = img.shape      
        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        
        self.model=model
        if self.model=='full':
            self.full_net.setInput(blob)
            output_layers_names = self.full_net.getUnconnectedOutLayersNames()
            layerOutputs =self.full_net.forward(output_layers_names)
            classes=['face','back']
        if self.model=='tiny':
            self.tiny_net.setInput(blob)
            output_layers_names = self.tiny_net.getUnconnectedOutLayersNames()
            layerOutputs =self.tiny_net.forward(output_layers_names)
            classes=['face']  
        boxes = []
        confidences = []
        class_ids = []
        for output in layerOutputs:
          for detection in output:
              scores = detection[5:]
              class_id = np.argmax(scores)
              confidence = scores[class_id]
              if confidence > 0.50:
                  center_x = int(detection[0]*width)
                  center_y = int(detection[1]*height)
                  w= int(detection[2]*width)
                  h= int(detection[3]*height)
                  x = int(center_x - (w/2))
                  y = int(center_y - (h/2))
                  boxes.append([x, y,h, w])
                  confidences.append((float(confidence)))
                  class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        face_box=[]
        conf=[] 
        if len(indexes)>0:
            for i in indexes.flatten():
                label = str(classes[class_ids[i]])
                if label==classes[0]:
                    face_box.append( boxes[i])
                    conf.append(confidences[i])
        return img,face_box,conf  
    def show_output(self,img,face_box,prediction_list=None,frame_status=False):
        self.img=img
        self.face_box=face_box
        self.prediction_list=prediction_list
        self.frame_status=frame_status
        font = cv2.FONT_HERSHEY_PLAIN
        font_=cv2.FONT_HERSHEY_COMPLEX
        color=(255,200,200)
        if len(face_box)>0:
            for i in range(len(face_box)):
              box=face_box[i]
              x,y,w,h=box[0],box[1],box[2],box[3]
              if prediction_list==None:
                label="{}".format('face')
              else:
                label = "{}".format(prediction_list[i])
              cv2.rectangle(self.img,(x,y), (x+h,y+w), (255,255,255), 3)
              (text_width, text_height) = cv2.getTextSize(label, font, fontScale=1, thickness=1)[0]
              text_offset_x,text_offset_y=x,y-10
              box_coords = ((text_offset_x, text_offset_y+10), (text_offset_x + text_width + 10, text_offset_y - text_height-10))
              cv2.rectangle(self.img, box_coords[0], box_coords[1], (0,0,0), cv2.FILLED)
              cv2.putText(self.img, label, (text_offset_x, text_offset_y), font, fontScale=1, color=color, thickness=1)

        if self.frame_status==False:
            img_output= cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB) # Converting BGR to RGB
            # from IPython.display import display
            return display(Image.fromarray(img_output))
        else:
            img_output=self.img
            return img_output