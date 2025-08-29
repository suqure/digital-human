import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasetsss import MyDataset
from syncnet import SyncNet_color
from unet import Model
import random
import torchvision.models as models

def get_args():
    parser = argparse.ArgumentParser(description='Train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_syncnet', action='store_true',required=False, default=True,  help="if use syncnet, you need to set 'syncnet_checkpoint'")
    parser.add_argument('--syncnet_checkpoint', type=str,required=False,  default="checkpoint/syncent.pth")
    parser.add_argument('--dataset_dir', type=list[str],required=False, default=["dataset"])
    parser.add_argument('--save_dir', type=str,required=False,  help="trained model save path.",default="checkpoint")
    parser.add_argument('--see_res',required=False, action='store_true',default=True)
    parser.add_argument('--epochs',required=False, type=int, default=100)
    parser.add_argument('--batchsize',required=False, type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--pretrain', type=str,required=False,default="checkpoint/unet.pth")

    return parser.parse_args()

args = get_args()
use_syncnet = args.use_syncnet
# Loss functions
class PerceptualLoss():
    
    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i,layer in enumerate(list(cnn)):
            model.add_module(str(i),layer)
            if i == conv_3_3_layer:
                break
        return model

    def __init__(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

def train(net, epoch, batch_size, lr):
    content_loss = PerceptualLoss(torch.nn.MSELoss())
    if use_syncnet:
        if args.syncnet_checkpoint == "":
            raise ValueError("Using syncnet, you need to set 'syncnet_checkpoint'.Please check README")
            
        syncnet = SyncNet_color().eval().cuda()
        syncnet.load_state_dict(torch.load(args.syncnet_checkpoint))
    save_dir= args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    dataloader_list = []
    dataset_list = []
    dataset_dir_list = args.dataset_dir
    for dataset_dir in dataset_dir_list:
        dataset = MyDataset(dataset_dir)
        train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=False, num_workers=4)
        dataloader_list.append(train_dataloader)
        dataset_list.append(dataset)
    
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.L1Loss()
    
    for e in range(epoch):
        net.train()
        random_i = random.randint(0, len(dataset_dir_list)-1)
        dataset = dataset_list[random_i]
        train_dataloader = dataloader_list[random_i]
        
        with tqdm(total=len(dataset), desc=f'Epoch {e + 1}/{epoch}', unit='img') as p:
            for batch in train_dataloader:
                imgs, labels, audio_feat = batch
                imgs = imgs.cuda()
                labels = labels.cuda()
                audio_feat = audio_feat.cuda()
                preds = net(imgs, audio_feat)
                if use_syncnet:
                    y = torch.ones([preds.shape[0],1]).float().cuda()
                    a, v = syncnet(preds, audio_feat)
                    sync_loss = cosine_loss(a, v, y)
                loss_PerceptualLoss = content_loss.get_loss(preds, labels)
                loss_pixel = criterion(preds, labels)
                if use_syncnet:
                    loss = loss_pixel + loss_PerceptualLoss*0.01 + 10*sync_loss
                else:
                    loss = loss_pixel + loss_PerceptualLoss*0.01
                p.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                p.update(imgs.shape[0])
                
        # if e % 2 == 0:
        torch.save(net.state_dict(), os.path.join(save_dir, str(e)+'_unet.pth'))
        if args.see_res:
            net.eval()
            img_concat_T, img_real_T, audio_feat = dataset.__getitem__(random.randint(0, dataset.__len__()))
            img_concat_T = img_concat_T[None].cuda()
            audio_feat = audio_feat[None].cuda()
            with torch.no_grad():
                pred = net(img_concat_T, audio_feat)[0]
            pred = pred.cpu().numpy().transpose(1,2,0)*255
            pred = np.array(pred, dtype=np.uint8)
            img_real = img_real_T.numpy().transpose(1,2,0)*255
            img_real = np.array(img_real, dtype=np.uint8)
            cv2.imwrite("./tmp/epoch_"+str(e)+".jpg", pred)
            cv2.imwrite("./tmp/epoch_"+str(e)+"_real.jpg", img_real)
    # torch.save(net.state_dict(), os.path.join(save_dir, str(e)+'_unet.pth'))    



def trainUnet(save_dir,dataset_dir_list, epoch,lr,syncent_cpk,pretrain,saveCount):
    best = None
    net = Model()
    if pretrain is not None and os.path.exists(pretrain):
        net.load_state_dict(torch.load(pretrain))
    content_loss = PerceptualLoss(torch.nn.MSELoss())
    if syncent_cpk is not None and  os.path.exists(syncent_cpk):
        syncnet = SyncNet_color().eval().cuda()
        syncnet.load_state_dict(torch.load(syncent_cpk))
        use_syncnet = True
    else:
        use_syncnet = False 
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    dataloader_list = []
    dataset_list = []
    net.cuda()   
    for dataset_dir in dataset_dir_list:
        dataset = MyDataset(dataset_dir)
        train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=False, num_workers=4)
        dataloader_list.append(train_dataloader)
        dataset_list.append(dataset)
    
    optimizer = optim.Adam(net.parameters(), lr=float(lr))
    criterion = nn.L1Loss()
    best = None
    result = []
    for e in range(epoch):
        net.train()
        random_i = random.randint(0, len(dataset_dir_list)-1)
        dataset = dataset_list[random_i]
        train_dataloader = dataloader_list[random_i]
        last_loss = None
        with tqdm(total=len(dataset), desc=f'Epoch {e + 1}/{epoch}', unit='img') as p:
            for batch in train_dataloader:
                imgs, labels, audio_feat = batch
                imgs = imgs.cuda()
                labels = labels.cuda()
                audio_feat = audio_feat.cuda()
                preds = net(imgs, audio_feat)
                if use_syncnet:
                    y = torch.ones([preds.shape[0],1]).float().cuda()
                    a, v = syncnet(preds, audio_feat)
                    sync_loss = cosine_loss(a, v, y)
                loss_PerceptualLoss = content_loss.get_loss(preds, labels)
                loss_pixel = criterion(preds, labels)
                if use_syncnet:
                    loss = loss_pixel + loss_PerceptualLoss*0.01 + 10*sync_loss
                else:
                    loss = loss_pixel + loss_PerceptualLoss*0.01 
                p.set_postfix(**{'loss (batch)': loss.item()})
                last_loss = loss.item()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                p.update(imgs.shape[0])
                
        # if e % 2 == 0:
        if best is None or best>last_loss:
            best = last_loss
            torch.save(net.state_dict(), os.path.join(save_dir, 'best_unet.pth'))
        if e % saveCount == 0:    
            torch.save(net.state_dict(), os.path.join(save_dir, str(e)+'_unet.pth'))
            result.append((str(epoch),loss.item()))
        if args.see_res:
            net.eval()
            img_concat_T, img_real_T, audio_feat = dataset.__getitem__(random.randint(0, dataset.__len__()))
            img_concat_T = img_concat_T[None].cuda()
            audio_feat = audio_feat[None].cuda()
            with torch.no_grad():
                pred = net(img_concat_T, audio_feat)[0]
            pred = pred.cpu().numpy().transpose(1,2,0)*255
            pred = np.array(pred, dtype=np.uint8)
            img_real = img_real_T.numpy().transpose(1,2,0)*255
            img_real = np.array(img_real, dtype=np.uint8)
            cv2.imwrite("./tmp/epoch_"+str(e)+".jpg", pred)
            cv2.imwrite("./tmp/epoch_"+str(e)+"_real.jpg", img_real) 
    result.append(('最佳',best))
    return result    

if __name__ == '__main__':
    
    
    net = Model()
    if os.path.exists(args.pretrain):
        net.load_state_dict(torch.load(args.pretrain))
    net.cuda()  
    train(net, args.epochs, args.batchsize, args.lr)