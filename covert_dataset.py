import os
import cv2, torch,shutil
import numpy as np 
from get_landmark import Landmark  
import torch.nn.functional as F 
from whisper import load_model as load_whisper
from whisper.audio import load_audio,log_mel_spectrogram 


class AudioEncoder(torch.nn.Module):
    def __init__(self,encoder):
        super().__init__()
        self.conv1 = encoder.conv1
        self.conv2 = encoder.conv2
        self.positional_embedding = encoder.positional_embedding
        self.blocks = encoder.blocks

    def forward(self, x: torch.Tensor):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = (x + self.positional_embedding).to(x.dtype)
        embeddings = [x]
        for block in self.blocks:
            x = block(x)
            embeddings.append(x)  
        embeddings = torch.stack(embeddings, axis=1)
        return embeddings

def extract_train_data(path,sr=16000,output_dir="dataset"):
    name,_ = os.path.splitext(os.path.basename(path))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    path = covert25Fps(path)
    full_body_dir = output_dir+"/raw"
    if not os.path.exists(full_body_dir):
        os.mkdir(full_body_dir)
    print("extract audio")
 
    out_wav = output_dir+"/%s.wav"%name
    cmd = f'ffmpeg -i {path} -f wav -ar {sr} {out_wav}'
    os.system(cmd)
    exract_whisper_feature(out_wav)
    print("extract image") 
    counter = 0
    cap = cv2.VideoCapture(path)
        
    print("extracting images...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(full_body_dir+"/"+str(counter)+'.jpg', frame)
        counter += 1
    exract_landmark(full_body_dir)
    print("extracting finish")
    
def exract_landmark(path):
    print("extracting landmarks ...")
    landmark = Landmark()
    landmarks_dir = path.replace(path.split("/")[-1], "landmark")
    if not os.path.exists(landmarks_dir):
        os.mkdir(landmarks_dir)
    for img_name in os.listdir(path):
        if not img_name.endswith(".jpg"):
            continue
        img_path = os.path.join(path, img_name)
        lms_path = os.path.join(landmarks_dir, img_name.replace(".jpg", ".lms"))
        pre_landmark, x1, y1 = landmark.detect(img_path)
        with open(lms_path, "w") as f:
            for p in pre_landmark:
                x, y = p[0]+x1, p[1]+y1
                f.write(str(x))
                f.write(" ")
                f.write(str(y))
                f.write("\n")

@torch.no_grad()
def exract_whisper_feature(path,device="cpu"): 
    print("Loading the whisper Model...")
    model = load_whisper("./checkpoint/tiny.pt",device=device)  
    model.eval()
    model = AudioEncoder(model.encoder)
    audio = load_audio(path)
    audio = torch.from_numpy(audio)  
    mel = log_mel_spectrogram(audio)
    num_sample=3000
    i = 0 
    samples = mel.shape[-1]
    embs = []
    while i< samples:
        end = min(i + num_sample,samples)
        array = mel[:,i:i+num_sample]
        if array.shape[-1] > num_sample:
                array = array.index_select(dim=-1, index=torch.arange(num_sample))
        if array.shape[-1] < num_sample:
                pad_widths = [(0, 0)] * array.ndim
                pad_widths[-1] = (0, num_sample - array.shape[-1])
                array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
        segment = array.unsqueeze(0)
        embeddings  = model(segment)
        embeddings = embeddings.transpose(1,2)
        embeddings = embeddings.squeeze(0)
        emb_end_idx = int((end - i) / 2)
        embs.append(embeddings[:emb_end_idx])
        i +=num_sample
    feats = torch.cat(embs,dim=0)
    feats = feature2chunks(feats)
    np.save(path.replace(path.split("/")[-1], 'whisper.npy'), feats.cpu().detach().numpy())
def feature2chunks(feature_array:torch.Tensor,fps:int=25,audio_feat_length=[2,2]):
        chunks = []
        idx_multiplier = 50.0 / fps
        i = 0  
        length = feature_array.shape[0]
        while True:
            start_idx = int(i * idx_multiplier)  
            selected_feature = [] 
            center_idx = int(i * 50 / fps)
            left_idx = center_idx - audio_feat_length[0] * 2
            right_idx = center_idx + (audio_feat_length[1] + 1) * 2

            for idx in range(left_idx, right_idx):
                idx = max(0, idx)
                idx = min(length - 1, idx)
                x = feature_array[idx]
                selected_feature.append(x) 
            selected_feature = torch.cat(selected_feature, dim=0)
            selected_feature = selected_feature.reshape(-1,384)  
            chunks.append(selected_feature)
            i += 1
            if start_idx > length:
                break
        feats = torch.stack(chunks)
        return feats
    
def covert25Fps(path,output_dir="tmp"):
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  
    os.mkdir(output_dir)     
    name,_ = os.path.splitext(os.path.basename(path))
    output = f'{output_dir}/{name}.mp4'
    cmd = f"ffmpeg -i {path} -r 25 {output}"
    os.system(cmd)
    return output

if __name__ == "__main__":
    print("start")   