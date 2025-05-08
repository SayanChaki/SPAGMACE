import os
import h5py
import numpy as np
import cv2

import torch

from gmair.config import config as cfg

class FruitDataset(torch.utils.data.Dataset):
    def __init__(self, root, anno_file):
        super().__init__()
        self.root = root
        self.info = anno_file
        self.imageFileNames = os.listdir(root)
        self.annoFileNames = os.listdir(anno_file)
        self.imageFileNames.sort()
        self.annoFileNames.sort()
        self.train_images = []
        self.anno = {}
        self.count = {}
        
        for f in self.imageFileNames:
            self.train_images.append(f)
        i = 0
        for f in self.annoFileNames:
            with open(os.path.join(self.info, f), "r") as fl:
                lines = fl.readlines()
                bbox = []
                for line in lines:
                    line = line.strip('\n')
                    datas = line.split(',')
                    if datas.__len__() == 1:
                        self.count[i] = (int(datas[0]))
                    else:
                        c=1
                        x, y, w, h = datas
                        bbox.append([float(x) - float(w)/2, float(y) - float(h)/2, 
                            float(w), float(h), float(c)])
                bbox = np.array(bbox)
                
                if self.count[i] > 100:
                    self.count[i] = 100
                    self.anno[i] = bbox[:100, :]
                else:
                    if self.count[i] == 0:
                        self.anno[i] = -np.ones((100, 5),dtype=int)
                        #print(f)
                    else:
                        self.anno[i] = np.row_stack((bbox, -np.ones((100 - self.count[i], 5),dtype=int)))
                i += 1    
                 

    def __getitem__(self, index):
        img_info = self.train_images[index]
        img_path = os.path.join(self.root, img_info)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)
        # image = image[..., None]
        image = np.moveaxis(image, -1, 0)
        image = image.astype(np.float32)
        image /= 255.0
        bbox = self.anno[index]
        count = self.count[index]
        print("THE SHAPE OF IMAGE IS")
        print(image.shape)
        return image, bbox, count
    



    '''def __getitem__(self, index):
      img_info = self.train_images[index]
      img_path = os.path.join(self.root, img_info)
      image = cv2.imread(img_path)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      # Rotate the image by an angle greater than 90 degrees (e.g., 120 degrees)
      angle = 120
      h, w = image.shape[:2]
      center = (w // 2, h // 2)
    
      # Compute the rotation matrix
      rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
      # Determine the new bounding box size after rotation
      cos_val = abs(rotation_matrix[0, 0])
      sin_val = abs(rotation_matrix[0, 1])
      new_w = int((h * sin_val) + (w * cos_val))
      new_h = int((h * cos_val) + (w * sin_val))

      # Adjust the rotation matrix to account for translation
      rotation_matrix[0, 2] += (new_w / 2) - center[0]
      rotation_matrix[1, 2] += (new_h / 2) - center[1]

      # Rotate the image and keep the background black
      image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

      # Convert to grayscale and apply a binary threshold to find non-black areas
      gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
      _, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

      # Find contours around the non-black area
      contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        image = image[y:y+h, x:x+w]  # Crop the image to the bounding box

      # Resize the cropped image to (128, 128) with 3 channels
      image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)

      # Move channels to the first dimension and normalize
      image = np.moveaxis(image, -1, 0)  # Convert to shape (3, 128, 128)
      image = image.astype(np.float32)
      image /= 255.0

      bbox = self.anno[index]
      count = self.count[index]
    
      print("THE SHAPE OF IMAGE IS")
      print(image.shape)
      return image, bbox, count'''


    def __len__(self):
        return len(self.train_images)
        
