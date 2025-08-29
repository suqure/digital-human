import os,json,cv2,librosa,subprocess,platform,torch,shutil
from torch import nn
import numpy as np
from unet import Model 
from time import time     
covert_path = "mnn" 
class Wav2LipModel(nn.Module):
    def __init__(self,  
                checkpoint_path,
                device='cpu'):
        super().__init__()
        self.device = device
        model = load_model(checkpoint_path,device=device) 
        model.to(device)
        model.eval()
        self.model = model
        self.mask = torch.ones(160,160,3)
        self.mask[5:150,5:155]=0
        
    def forward(self,
                image:torch.Tensor,
                audio:torch.Tensor):
        y = audio.unsqueeze(0)  
        mask = self.mask * image 
        x = torch.cat((image,mask),dim=2).unsqueeze(0).permute((0, 3, 1, 2))
        pred = self.model(x,y)[0].permute((1,2,0))*255  
        return pred


def load_model(path,device="cpu"):
    model = Model()
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    model.eval()
    return model
def read_landmarks(path,x,y):
    lms_list = []
    with open(path, "r") as f:
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
    if xmax>x:
        xmax = x
    width = xmax - xmin
    ymax = ymin + width
    if ymax>y:
        ymax = y
    return ymin,ymax,xmin,xmax
 



def loadFrame(path,lenght=16000):
    files = os.listdir(path)
    files.sort(key=lambda x:int(x.split(".")[0]))
    frames = []
    names = []
    coords = []
    landmarks_dir = path.replace(path.split("/")[-1], "landmark")
    for name in files: 
        img = cv2.imread(path+"/"+name)
        frames.append(img)
        names.append(name)
        lms_path = os.path.join(landmarks_dir, name.replace(".jpg", ".lms"))
        coord = read_landmarks(lms_path,img.shape[1],img.shape[0])
        coords.append(coord)
        if len(frames)>=lenght:
            break
    return frames,names,coords
 
def loadWav(path):
    wavform,_ = librosa.load(path,sr=16000)
    audio = torch.from_numpy(wavform)
    return audio
 

def exportOnnx(checkpoint_path):
    img = torch.randn((160,160,3)) 
    model =  Wav2LipModel(checkpoint_path=checkpoint_path) 
    feat = torch.randn((25,24,32)) 
    torch.onnx.export(model, (img,feat), 'model.onnx', input_names=['frame', 'mel'], output_names=['output'],opset_version=11)    

def exportModel(checkpoint_path,data_dir,frameCount):
    if os.path.exists(covert_path):
       shutil.rmtree(covert_path)
    os.mkdir(covert_path)  
    exportOnnx(checkpoint_path)
    cmd = f"MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn    --bizCode biz "
    os.system(cmd) 
    shutil.move("model.mnn",covert_path+"/model.mnn")
    exportConfig(data_dir,needSilence=True,count=frameCount)
    shutil.move("config.json",covert_path+"/config.json")
    shutil.copyfile("checkpoint/whisper.pt",covert_path+"/feature.pt")
    os.mkdir(covert_path+"/raw")
    files = os.listdir(data_dir)
    files.sort(key=lambda x:int(x.split(".")[0]))
    num = 0
    for name in files: 
        shutil.copyfile(data_dir+"/%s"%name,covert_path+"/raw/%s"%name)
        num+=1
        if num>=frameCount:
            break
        
    

def zipFile(image:None,name:str='角色模型',desc:str="数字人角色模型"):
    print("start package model")
    fileName = "%s.zip"%(int(time()))
    cover = ""    
    path = "zip" 
    if os.path.exists(path):
        shutil.rmtree(path)
    shutil.move(covert_path,path)
    if image is not None:
        cover = "cover.jpg"
        shutil.copyfile(image,path+"/"+cover)
    config = '''
{
    "id": %s,
    "type": 1,
    "name": "%s",
    "cover": "%s",
    "description":"%s"
}
    '''%(int(time()),name,cover,desc)
    with open('live2d.config', 'w',encoding='utf-8') as f:
        f.write(config)
    shutil.move("live2d.config",path+"/live2d.config")
    shutil.make_archive('model','zip',path)
    shutil.move("model.zip",fileName)
    print("end package model")
    return fileName

def exportConfig(path,needSilence=False,type=2,ranges=[],count=1500):
    frames,_,coords = loadFrame(path,count) 
    faces =  {}
    frame_h, frame_w = frames[0].shape[:-1]
    for i,m in enumerate(coords): 
        y1, y2, x1, x2 = coords[i]
        faces[str(i)] = [int(y1),int(y2), int(x1), int(x2)] 
    config = {
        "type": type,
        "needSilence":needSilence,
        "faces":faces,
        "width":frame_w,
	    "height":frame_h,
        "ranges":ranges
    }
    with open("config.json","w", encoding='utf-8') as f: ## 设置'utf-8'编码
        f.write(json.dumps(config,ensure_ascii=False)) 
 
@torch.inference_mode() 
def predicate(wav ,checkPoint,framePath):
    output = "output/%s.mp4"%str(int(time()))
    model =  Wav2LipModel(checkPoint) 
    fps = 25 
    feature = torch.load("checkpoint/whisper.pt",map_location="cpu")
    audio = loadWav(wav) 
    feats = feature(audio)
    frames,_,coords = loadFrame(framePath,lenght=1600)
    full_frames = frames[:len(feats)]
    frame_h, frame_w = full_frames[0].shape[:-1]
    out = cv2.VideoWriter('tmp/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(wav, 'tmp/result.avi',output)
    for i, feat in enumerate(feats): 
        y = feat 
        img = full_frames[i]
        y1,y2,x1,x2 =coords[i]
        crop_img = img[y1:y2, x1:x2]
        crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
        x = torch.from_numpy(crop_img)[4:164,4:164]
        t1 = time()
        x = x/255.0
        preds = model(x,y).cpu().numpy()
        t2 = time()
        print("excute forward cos:%.3f" % ( t2 - t1)) 
        p = preds.astype(np.uint8)
        crop_img[4:164,4:164] = p
        p = cv2.resize(crop_img, (x2 - x1, y2 - y1)) 
        img[y1:y2, x1:x2] = p 
        out.write(img) 
    out.release()
    subprocess.call(command, shell=platform.system() != 'Windows')
    os.remove('tmp/result.avi')
    return output   
     
if __name__ == '__main__':
    print("==============start") 


     