from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms as trans
import torchvision
from algorithm.model import MobileFaceNet, PRCNN, HopeNet, ShuffledHopeNet, quantize_model
from algorithm.model.shufflehopenet import ShuffledHopeNet
from algorithm.model.hopenetlite import HopeNetLite



# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
facebank_path='resources/face_lib'
faces_dir='resources/faces'
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
mobilefacenet_qint8 = quantize_model(mobilefacenet)
mobilefacenet_qint8.to(device)
mobilefacenet_qint8.eval()
def prepare_facebank(facebank_path, model, force_rebuild=False):
    """
    准备特征库的核心方法，支持批量处理
    """
    facebank_file = Path(facebank_path) / 'facebank.pth'
    names_file = Path(facebank_path) / 'names.npy'

    if not force_rebuild and facebank_file.exists() and names_file.exists():
        targets = torch.load(facebank_file, map_location=device)
        names = np.load(names_file)
        return targets, names

    embeddings = []
    name_list = ['Unknown']

    # 批量处理每个人物的图像
    for person_dir in Path(facebank_path).iterdir():
        if not person_dir.is_dir() or person_dir.name.startswith('.'):
            continue

        # 批量读取图像
        img_batch = []
        valid_files = []
        for img_file in person_dir.glob('*.*'):
            if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                img = cv2.imread(str(img_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_batch.append(img)
                valid_files.append(img_file)

        if not img_batch:
            continue

        # 批量处理原始图像和镜像图像
        with torch.no_grad():
            # 处理原始图像
            orig_tensors = torch.stack([mobilefacenet_transform(img) for img in img_batch]).to(device)
            orig_embs = model(orig_tensors)

            # 处理镜像图像
            mirror_imgs = [cv2.flip(img, 1) for img in img_batch]
            mirror_tensors = torch.stack([mobilefacenet_transform(img) for img in mirror_imgs]).to(device)
            mirror_embs = model(mirror_tensors)

            # 融合特征并归一化
            fused_embs = l2_norm(orig_embs + mirror_embs)

            # 计算平均特征并二次归一化
            avg_emb = torch.mean(fused_embs, dim=0, keepdim=True)
            avg_emb = l2_norm(avg_emb)

            embeddings.append(avg_emb)
            name_list.append(person_dir.name)

    # 保存特征库
    targets = torch.cat(embeddings) if embeddings else torch.Tensor()
    names = np.array(name_list)

    torch.save(targets, facebank_file)
    np.save(names_file, names)

    return targets, names
# 准备特征库
face_targets, face_names = prepare_facebank(facebank_path, mobilefacenet)

prcnn = PRCNN(image_size=160, thresholds=[0.8, 0.9],min_face_size=40,pnet_path=pnet_path, rnet_path=rnet_path, device=device).to(device)
# prcnn.load_state_dict(torch.load(prcnn_path, map_location=device))
prcnn.eval()

hopenet = HopeNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
hopenet.load_state_dict(torch.load(hopenet_path, map_location=device))
hopenet = hopenet.to(device)  # 替换model.cuda()
hopenet.eval()

shuffledhopenet = ShuffledHopeNet([4, 8, 4], [24, 116, 232, 464, 1024])
shuffledhopenet.load_state_dict(torch.load(shuffledhopenet_path, map_location=device), strict=False)
shuffledhopenet.to(device)
shuffledhopenet.eval()

hopenetlite = HopeNetLite()
saved_state_dict = torch.load(hopenetlite_path, map_location="cpu")
hopenetlite.load_state_dict(saved_state_dict, strict=False)
hopenetlite.to(device)  # 替换model.cuda()
hopenetlite.eval()

hopenetlite_qint8 = quantize_model(hopenetlite)
hopenetlite_qint8.to(device)
hopenetlite_qint8.eval()

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