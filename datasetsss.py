import os
import cv2
import torch
import random
import numpy as np
import random

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MyDataset(Dataset):
    
    def __init__(self, data_dir):
    
        self.img_path_list = []
        self.lms_path_list = []
        self.feats = []
        self.feats_idx = [] 
        if isinstance(data_dir, list):
            for dir in data_dir:
                imgs,lms,feats = self.load_data(dir)
                self.img_path_list.extend(imgs)
                self.lms_path_list.extend(lms)
                self.feats.append(feats)
                self.feats_idx.append(len(self.img_path_list))
        else:
            imgs,lms,feats = self.load_data(data_dir)
            self.img_path_list.extend(imgs)
            self.lms_path_list.extend(lms)
            self.feats.append(feats)
            self.feats_idx.append(len(self.img_path_list))
        print(data_dir) 
        print(len(self.img_path_list))
        
    def __len__(self):
        return len(self.img_path_list)
    
    def get_features(self,index):
        offset = 0
        for i, m in enumerate(self.feats_idx):
            if index<m:
                return self.feats[i],offset
            offset = m
        return self.feats[-1],offset

    def load_data(self,data_dir):
        img_paths = []
        lms_paths = []
        for i in range(len(os.listdir(data_dir+"/raw/"))):

            img_path = os.path.join(data_dir+"/raw/", str(i)+".jpg")
            lms_path = os.path.join(data_dir+"/landmark/", str(i)+".lms")
            img_paths.append(img_path)
            lms_paths.append(lms_path)  
        audio_feats = np.load(data_dir+"/whisper.npy")
        return img_paths,lms_paths,audio_feats
    
    def get_audio_features(self, features, index):
        feats,offset=features
        index = index - offset
        return torch.from_numpy(feats[index])
        
    
    
    def process_img(self, img, lms_path, img_ex, lms_path_ex):

        lms_list = []
        with open(lms_path, "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                arr = line.split(" ")
                arr = np.array(arr, dtype=np.float32)
                lms_list.append(arr)
        lms = np.array(lms_list, dtype=np.int32)
        xmin = lms[1][0]
        ymin = lms[52][1]
        if xmin<0:
            xmin = 0
        if ymin<0:
            ymin = 0
        
        xmax = lms[31][0]
        if xmax>img.shape[1]:
            xmax = img.shape[1]
        width = xmax - xmin
        ymax = ymin + width
        if ymax>img.shape[0]:
            ymax = img.shape[0]
        
        crop_img = img[ymin:ymax, xmin:xmax]
        crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
        img_real = crop_img[4:164, 4:164].copy()
        img_real_ori = img_real.copy()
        img_masked = cv2.rectangle(img_real,(5,5,150,145),(0,0,0),-1)
        
        lms_list = []
        with open(lms_path_ex, "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                arr = line.split(" ")
                arr = np.array(arr, dtype=np.float32)
                lms_list.append(arr)
        lms = np.array(lms_list, dtype=np.int32)
        xmin = lms[1][0]
        ymin = lms[52][1]
        if xmin<0:
            xmin = 0
        if ymin<0:
            ymin = 0
        
        xmax = lms[31][0]
        if xmax>img.shape[1]:
            xmax = img.shape[1]

        width = xmax - xmin
        ymax = ymin + width
        if ymax>img.shape[0]:
            ymax = img.shape[0]
        crop_img = img_ex[ymin:ymax, xmin:xmax]
        crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
        img_real_ex = crop_img[4:164, 4:164].copy()
        
        img_real_ori = img_real_ori.transpose(2,0,1).astype(np.float32)
        img_masked = img_masked.transpose(2,0,1).astype(np.float32)
        img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)
        
        img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
        img_real_T = torch.from_numpy(img_real_ori / 255.0)
        img_masked_T = torch.from_numpy(img_masked / 255.0)
        img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)

        return img_concat_T, img_real_T

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path_list[idx])
        lms_path = self.lms_path_list[idx]
        
        ex_int = random.randint(0, self.__len__()-1)
        img_ex = cv2.imread(self.img_path_list[ex_int])
        lms_path_ex = self.lms_path_list[ex_int]
        
        img_concat_T, img_real_T = self.process_img(img, lms_path, img_ex, lms_path_ex)
        
        audio_feat = self.get_audio_features(self.get_features(idx), idx) 
         
        audio_feat = audio_feat.reshape(25,24,32)
        
        return img_concat_T, img_real_T, audio_feat



if __name__ == "__main__":
    print("execute")
