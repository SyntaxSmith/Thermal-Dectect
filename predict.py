# %%
test_data_path='data/test'

# %% [markdown]
# 读入文件夹列表，每个文件夹包含一个帧序列，使用模型进行目标检测，将结果保存在同名txt文件中，文本内容形如：
# {“res”:[[x,y,width,height],[x,y,width,height],[],...[x,y,width,height]}

# %%
import os

from ultralytics import YOLO

model=YOLO('runs/detect/train/weights/best.pt')

# %%
import json
import PIL


def location_transfer(img,pre):
    # 将yolo给出的预测结果转换为原图的坐标
    # 坐标格式为：左上角x，左上角y，宽，高，使用绝对像素值
    img=PIL.Image.open(img)
    size=img.size
    pre_new=[]
    x_left=int(pre[0]-pre[2]/2)
    y_left=int(pre[1]-pre[3]/2)
    img.close()
    return [x_left,y_left,int(pre[2]),int(pre[3])]
    
folder_list = os.listdir(test_data_path)
folder_list.sort()
model=YOLO('runs/detect/train/weights/best.pt')
def predictor():
    for folder in folder_list:
        file_list = os.listdir(test_data_path+'/'+folder)
        print(file_list)
        # 剔除json文件，按照文件名排序
        if 'IR_label.json' in file_list:
            file_list.remove('IR_label.json')
        file_list.sort()
        file_list=[test_data_path+'/'+folder+'/'+file for file in file_list]
        res=[]
        # 载入第一帧的标注
        res.append(json.load(open(os.path.join(test_data_path,folder_list[0],'IR_label.json')))['res'][0])
        for img in file_list[1:]:
            # 预测
            pre=model.predict(img)[0]
            # 转换坐标
            pre=pre.boxes.xywh.tolist()
            if len(pre)==0:
                res.append([])
            else:
                pre=pre[0]
                pre=location_transfer(img,pre)
                res.append(pre)
            del pre
        res={'res':res}
        with open(os.path.join('res',folder+'.txt'),'w+') as f:
            f.write(json.dumps(res))

# %%

predictor()


