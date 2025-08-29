import gradio as gr 
import os,time
from covert_dataset import extract_train_data
from syncnet import train as syn_train
from train import trainUnet 
from model import predicate,exportModel,zipFile
 
def checkDatasetDir(file_dir):
    raw = file_dir+"/raw"
    landmark =  file_dir+"/landmark"
    feature =  file_dir+"/whisper.npy"
    if os.path.exists(landmark) and os.path.exists(raw) and os.path.exists(feature):
        return True 
    return False

def refreshDataset(file_dir="dataset"):
    dataSetList = []
    dir_list = os.listdir(file_dir)
    for cur_file in dir_list:
        path = os.path.join(file_dir, cur_file)
        if checkDatasetDir(path):
            dataSetList.append(cur_file)
    return dataSetList
def refreshSynCpk():
    checkList = []
    dir_list = os.listdir("model") 
    for cur_dir in dir_list:
        path = os.path.join('model', cur_dir)
        file_list = os.listdir(path)
        for cur_file in file_list:
             if cur_file.endswith(".pth") and 'syncent' in cur_file:
                  checkList.append(os.path.join(path,cur_file))
    return checkList
def refreshCpk():
    checkList = []
    dir_list = os.listdir("model") 
    for cur_dir in dir_list:
        path = os.path.join('model', cur_dir)
        file_list = os.listdir(path)
        for cur_file in file_list:
             if cur_file.endswith(".pth") and 'unet' in cur_file:
                  checkList.append(os.path.join(path,cur_file))
    return checkList
         
             

def process_covert(video):
    name,_ = os.path.splitext(os.path.basename(video))
    output_dir = "dataset/%s"%name 
    try:
         extract_train_data(video,output_dir=output_dir) 
         return refreshDataset()
    except Exception as e:
            raise gr.Error(e)
def process_syn_train(dataset,epoch=100,lr=0.01,checkpoint=None,saveCpk=5): 
    output_dir = "model/%s"%str(int(time.time())) 
    dataset_dir = []
    for dir in dataset:
         dataset_dir.append("dataset/"+dir)
    try:
        return syn_train(output_dir, dataset_dir,epoch,lr,checkpoint,saveCpk) 
    except Exception as e: 
            raise gr.Error(e) 
def process_train(dataset,epoch=100,lr=0.01,synCheckPoint=None,checkpoint=None,saveCpk=5):
    output_dir = "model/%s"%str(int(time.time())) 
    dataset_dir = []
    for dir in dataset:
         dataset_dir.append("dataset/"+dir)
    try: 
         return trainUnet(output_dir,dataset_dir, epoch, lr,synCheckPoint,checkpoint,saveCpk) 
    except Exception as e:
            raise gr.Error(e)
def process(wav,dataset,checkpoint): 
    dataset_dir =  "dataset/%s/raw"%dataset
    try: 
         return predicate(wav, checkpoint, dataset_dir) 
    except Exception as e:
            raise gr.Error(e)
def process_model(dataset,checkpoint,timeCount,name,cover,desc): 
    dataset_dir =  "dataset/%s/raw"%dataset
    try: 
        exportModel(checkpoint, dataset_dir,timeCount*25) 
        return zipFile(image=cover,name=name,desc=desc)
    except Exception as e:
            raise gr.Error(e)

with gr.Blocks() as demo:
    gr.Markdown(
                """ 
                    <img src='/file=res/logo.png' style="float:left;"/> 
                    <div style="height:40px; "></div> 
                    <h1>小链数字人模型WebUI</h1>
                """
            )
    with gr.Tab("转换训练集"):
        with gr.Row():
            with gr.Column():
                video = gr.Video(label="视频")
                btn = gr.Button("转换训练数据集", variant="primary")
            with gr.Column(): 
                dataSetOut = gr.Dropdown(multiselect=True, choices=refreshDataset(), allow_custom_value=True, label="训练集") 
    with gr.Tab("唇形同步模型训练"):
        with gr.Row():
            with gr.Column():
                synEpoch = gr.Slider(minimum=10,maximum=200,value=100, label="训练轮数",step=1)
                synSave = gr.Slider(minimum=1,maximum=100,value=5, label="检查点轮数",step=1)
                synLr = gr.Textbox(value=0.01,label="学习率")
                synDataset = gr.Dropdown(multiselect=True, choices=refreshDataset(), allow_custom_value=True, label="训练集")
                synCheck = gr.Dropdown(multiselect=False,choices=refreshSynCpk(),  label="检查点") 
                synBtn = gr.Button("开始唇训练", variant="primary")
            with gr.Column(): 
                synTrainRs = gr.DataFrame(label='训练结果',headers=['训练轮数','损失率'])
    with gr.Tab("数字人模型训练"):
        with gr.Row():
            with gr.Column():
                epoch =  gr.Slider(minimum=10,maximum=400,value=200, label="训练轮数",step=1)
                unetSave = gr.Slider(minimum=1,maximum=100,value=5, label="检查点轮数",step=1)
                lr = gr.Textbox(value=0.01,label="学习率")
                dataset = gr.Dropdown(multiselect=True, choices=refreshDataset(), allow_custom_value=True, label="训练集")
                synModel = gr.Dropdown(multiselect=False,choices=refreshSynCpk(),  label="唇形同步模型")
                checkPoint = gr.Dropdown(multiselect=False,choices=refreshCpk(),  label="检查点")
                trainBtn = gr.Button("开始数字人训练", variant="primary")
            with gr.Column(): 
                trainRs = gr.DataFrame(label='训练结果',headers=['训练轮数','损失率'])
    with gr.Tab("模型推理测试"):
        with gr.Row():
            with gr.Column():
                wav = gr.Audio(label="声音",type='filepath') 
                dataFrame = gr.Dropdown(multiselect=False, choices=refreshDataset(), allow_custom_value=True, label="训练集")
                pretain = gr.Dropdown(multiselect=False,choices=refreshCpk(),  label="检查点")
                predBtn = gr.Button("开始推理", variant="primary")
            with gr.Column(): 
                pred = gr.Video(label='推理结果')
    with gr.Tab("模型转换"):
        with gr.Row():
            with gr.Column(): 
                dataFrame = gr.Dropdown(multiselect=False, choices=refreshDataset(), allow_custom_value=True, label="训练集")
                timeCount =  gr.Slider(minimum=5,maximum=60,value=10, label="视频时长(秒)",step=1)
                unet = gr.Dropdown(multiselect=False,choices=refreshCpk(),  label="检查点")
                name = gr.Textbox(value="角色模型",label="模型名称")
                cover = gr.Image(label="封面图片",type='filepath')  
            with gr.Column(): 
                desc = gr.Textbox(value="数字人模型",label="模型描述")
                model = gr.File(label='转换结果')
                convertBtn = gr.Button("开始转换", variant="primary")            
             
    btn.click(process_covert,[video],outputs=dataSetOut)
    synBtn.click(process_syn_train,[synDataset,synEpoch,synLr,synCheck,synSave],outputs=synTrainRs)
    trainBtn.click(process_train,[dataset,epoch,lr,synModel,checkPoint,unetSave],outputs=trainRs)
    predBtn.click(process,[wav,dataFrame,pretain],outputs=pred)
    convertBtn.click(process_model,[dataFrame,unet,timeCount,name,cover,desc],outputs=model)

if __name__ == "__main__":    
    demo.launch(server_name="0.0.0.0",inbrowser=True,allowed_paths=['./res'])  
