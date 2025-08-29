import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import cv2
import os
import numpy as np
from torch import optim
import random
import argparse



class Dataset(object):
    def __init__(self, dataset_dir):
        
        self.img_path_list = []
        self.lms_path_list = []
        self.feats = []
        self.feats_idx = [] 
        if isinstance(dataset_dir, list):
            for dir in dataset_dir:
                imgs,lms,feats = self.load_data(dir)
                self.img_path_list.extend(imgs)
                self.lms_path_list.extend(lms)
                self.feats.append(feats)
                self.feats_idx.append(len(self.img_path_list))
        else:
            imgs,lms,feats = self.load_data(dataset_dir)
            self.img_path_list.extend(imgs)
            self.lms_path_list.extend(lms)
            self.feats.append(feats)
            self.feats_idx.append(len(self.img_path_list)-1)
        print(dataset_dir) 
        print(len(self.img_path_list))


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
    def get_features(self,index):
        offset = 0
        for i, m in enumerate(self.feats_idx):
            if index<m:
                return self.feats[i],offset
            offset = m
        return self.feats[-1],offset     
    def __len__(self):

        return len(self.img_path_list)
    

    def get_audio_features(self, features, index,step=8): 
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
        img_real_ori = img_real_ori.transpose(2,0,1).astype(np.float32)
        img_real_T = torch.from_numpy(img_real_ori / 255.0)
        
        return img_real_T

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path_list[idx])
        lms_path = self.lms_path_list[idx]
        ex_int = random.randint(0, self.__len__()-1)
        img_ex = cv2.imread(self.img_path_list[ex_int])
        lms_path_ex = self.lms_path_list[ex_int]
        
        img_real_T = self.process_img(img, lms_path, img_ex, lms_path_ex)
        audio_feat = self.get_audio_features(self.get_features(idx), idx)  
        
        audio_feat = audio_feat.reshape(25,24,32) 
        y = torch.ones(1).float()
        
        return img_real_T, audio_feat, y

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class SyncNet_color(nn.Module):
    def __init__(self):
        super(SyncNet_color, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(3, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)
        
         
        self.audio_encoder = nn.Sequential(
            Conv2d(25, 256, kernel_size=3, stride=1, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(256, 256, kernel_size=3, stride=(2, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 256, kernel_size=3, stride=2, padding=2),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=(2,1)),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, face_sequences, audio_sequences): # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)
        
        return audio_embedding, face_embedding

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss
    
def train(save_dir, dataset_dir, epochs,lr,checkpoint,saveCount):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    train_dataset = Dataset(dataset_dir)
    train_data_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True,
        num_workers=4)
    model = SyncNet_color()
    if checkpoint is not None and os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint))
    model.cuda()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=float(lr))
    best = None
    result = []
    for epoch in range(epochs):
        for batch in train_data_loader:
            imgT, audioT, y = batch
            imgT = imgT.cuda()
            audioT = audioT.cuda()
            y = y.cuda()
            audio_embedding, face_embedding = model(imgT, audioT)
            loss = cosine_loss(audio_embedding, face_embedding, y)
            loss.backward()
            optimizer.step()
        print(epoch, loss.item())
        
        if best is None or best>loss.item():
            best = loss.item()  
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_syncent.pth'))
        if (epoch+1) % saveCount == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, str(epoch)+'_syncent.pth'))
            result.append((str(epoch),loss.item()))
    result.append(('最佳',best))
    return result
            
            
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str,required=False,default="checkpoint")
    parser.add_argument('--dataset_dir', type=list[str],required=False, default=["dataset"])
    parser.add_argument('--asr', type=str,required=False,default="whisper")
    parser.add_argument('--pretrain', type=str,required=False,default="checkpoint/syncent.pth")
    parser.add_argument('--lr', type=float,required=False,default=0.001)
    parser.add_argument('--epochs',required=False, type=int, default=100)
    opt = parser.parse_args()
     
    train(opt.save_dir, opt.dataset_dir,opt.epochs,opt.lr,opt.pretrain)
     
