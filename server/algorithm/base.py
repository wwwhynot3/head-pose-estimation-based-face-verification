import os.path
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import ImageFont, ImageDraw, Image
from torchvision import transforms as trans
import torchvision

from algorithm.model.mobilefacenet import l2_norm, MobileFaceNet
from algorithm.model.prcnn import PRCNN
from algorithm.model.quantization import quantize_model
# from algorithm.model import MobileFaceNet, PRCNN, HopeNet, ShuffledHopeNet, quantize_model
from algorithm.model.hopenet import HopeNet
from algorithm.model.shufflehopenet import ShuffledHopeNet
from algorithm.model.hopenetlite import HopeNetLite



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
facebank_path='resources/facebank'
faces_dir='resources/faces'
facebank_file_name = 'facebank.pth'
facebank_name_list_name = 'name_list.npy'
facebank_default_account = 'default'
output_dir='resources/results'
mobilefacenet_path = 'resources/model/mobilefacenet.pt'
# mobilefacenet_path = 'resources/model/model_mobilefacenet.pth'
# mobilefacenet_path = 'resources/model/face_recognition_mv.pkl'
pnet_path = 'resources/model/pnet.pt'
rnet_path = 'resources/model/rnet.pt'
hopenet_path = 'resources/model/hopenet.pt'
shuffledhopenet_path = 'resources/model/shuffledhopenet.pt'
hopenetlite_path = 'resources/model/hopenetlite.pt'

# 加载模型
mobilefacenet = MobileFaceNet(512).to(device)
mobilefacenet.load_state_dict(torch.load(mobilefacenet_path, map_location=device))
# mobilefacenet = quantize_model(mobilefacenet)
mobilefacenet.eval()

# mobilefacenet_qint8 = MobileFaceNet(512).to(device)
# mobilefacenet_qint8.load_state_dict(torch.load(mobilefacenet_path, map_location=device))
# mobilefacenet_qint8 = quantize_model(mobilefacenet)
# mobilefacenet_qint8.to(device)
# mobilefacenet_qint8.eval()


prcnn = PRCNN(image_size=160, thresholds=[0.98, 0.99],min_face_size=80,pnet_path=pnet_path, rnet_path=rnet_path, device=device).to(device)
# prcnn.load_state_dict(torch.load(prcnn_path, map_location=device))
prcnn.eval()
#
# prcnn_qint8 = PRCNN(image_size=160, thresholds=[0.8, 0.9],min_face_size=40,pnet_path=pnet_path, rnet_path=rnet_path, device=device).to(device).quantize().eval()

hopenet = HopeNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
hopenet.load_state_dict(torch.load(hopenet_path, map_location=device))
hopenet = hopenet.to(device)  # 替换model.cuda()
hopenet.eval()

# hopenet_qint8 = HopeNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
# hopenet_qint8.load_state_dict(torch.load(hopenet_path, map_location=device))
# hopenet_qint8 = quantize_model(hopenet_qint8.to(device))
# hopenet_qint8.to(device)
# hopenet_qint8.eval()


shuffledhopenet = ShuffledHopeNet([4, 8, 4], [24, 116, 232, 464, 1024])
shuffledhopenet.load_state_dict(torch.load(shuffledhopenet_path, map_location=device), strict=False)
shuffledhopenet.to(device)
shuffledhopenet.eval()

hopenetlite = HopeNetLite()
saved_state_dict = torch.load(hopenetlite_path, map_location="cpu")
hopenetlite.load_state_dict(saved_state_dict, strict=False)
hopenetlite.to(device)  # 替换model.cuda()
hopenetlite.eval()

# hopenetlite_qint8 = HopeNetLite()
# saved_state_dict = torch.load(hopenetlite_path, map_location="cpu")
# hopenetlite_qint8.load_state_dict(saved_state_dict, strict=False)
# hopenetlite_qint8 = quantize_model(hopenetlite_qint8.to(device))
# hopenetlite_qint8.to(device)
# hopenetlite_qint8.eval()

# 图像预处理（保持与训练时一致）
mobilefacenet_transform = trans.Compose([
    trans.ToPILImage(),
    trans.Resize((112, 112)),  # 单次resize即可
    trans.ToTensor(),
    trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

hopenet_transform = trans.Compose([
        trans.ToPILImage(),
        trans.Resize(224),
        trans.CenterCrop(224),
        trans.ToTensor(),
        trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def prepare_facebank(facebank_path, model, force_rebuild=False):
    """
    准备特征库的核心方法，支持批量处理
    """
    facebank_file = Path(facebank_path) / facebank_file_name
    names_file = Path(facebank_path) / facebank_name_list_name
    if not force_rebuild and facebank_file.exists() and names_file.exists():
        targets = torch.load(facebank_file, map_location=device)
        names = np.load(names_file)
        return targets, names

    embeddings = []
    name_list = ['Unknown']
    try:
    # 批量处理每个人物的图像
        for person in Path(facebank_path).iterdir():
            if person.is_dir() or person.suffix == '.pth' or person.suffix == '.npy':
                continue
            img = cv2.imread(str(person))
            name_list.append(person.stem)
            # 批量处理原始图像和镜像图像
            with torch.no_grad():
                # 处理原始图像
                orig_tensors = mobilefacenet_transform(img).to(device).unsqueeze(0)
                mirror_imgs = cv2.flip(img, 1)
                mirror_tensors = mobilefacenet_transform(mirror_imgs).to(device).unsqueeze(0)

                orig_embs = model(orig_tensors)
                mirror_embs = model(mirror_tensors)
                # 融合特征并归一化
                fused_embs = l2_norm(orig_embs + mirror_embs)

                # 计算平均特征并二次归一化
                avg_emb = torch.mean(fused_embs, dim=0, keepdim=True)
                avg_emb = l2_norm(avg_emb)

                embeddings.append(avg_emb)
                # embeddings.append(orig_embs)
    except Exception as e:
        traceback.print_exc()
        raise e
    # 保存特征库
    targets = torch.cat(embeddings) if embeddings else torch.Tensor()
    names = np.array(name_list)

    torch.save(targets, facebank_file)
    np.save(names_file, names)
    # print(f'{targets}')
    # print(f'{names}')
    return targets, names

def init_facebank():
    # 从resources/face_lib路径中有多个文件夹，每个文件夹是一个account, 文件夹下是用户上传的人脸，还有用户的facebank
    # 打开facebank_path下的所有目录
    facebank_map = dict()
    for facebank in Path(facebank_path).iterdir():
        if facebank.is_dir() and not facebank.name.startswith('.'):
            # 读取facebank.pth文件
            facebank_file = facebank / facebank_file_name
            if facebank_file.exists():
                targets = torch.load(facebank_file, map_location=device)
                # 读取names.npy文件
            else:
                raise FileNotFoundError(f'File {facebank_file} not found.')
            names_file = facebank / facebank_name_list_name
            if names_file.exists():
                names = np.load(names_file)
            else: #抛出异常
                raise FileNotFoundError(f'File {names_file} not found.')
            facebank_map[facebank.name] = (targets, names)
    return facebank_map

def cv2PutChineseText(img, text, position, textColor=(0, 0, 255), textSize=15):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    # 字体的格式

    # 绘制文本
    draw.text(position, text, textColor, font=font_style)
    # 转换回OpenCV格式
    return np.asarray(img)

facebank_map = init_facebank()
font_size = 25
font_style = ImageFont.truetype(
    "resources/fonts/SimSun.ttf", font_size, encoding="utf-8")
# face_targets, face_names = facebank_map['1']
# face_targets, face_names = prepare_facebank(facebank_path + '/1', mobilefacenet, force_rebuild=True)
# prepare_facebank(facebank_path + '/zhengkai', mobilefacenet, force_rebuild=True)

