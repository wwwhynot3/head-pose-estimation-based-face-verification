import torch
from torchvision import transforms as trans
from algorithm.model import MobileFaceNet, PRCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
facebank_path='resources/face_lib'
faces_dir='resources/faces'
output_dir='resources/results'
model_path = 'resources/model/MobileFace_Net'
# 加载模型
mobilefacenet = MobileFaceNet(512).to(device)
mobilefacenet.load_state_dict(torch.load(model_path, map_location=device))
mobilefacenet.eval()

prcnn = PRCNN(image_size=160, thresholds=[0.8, 0.9], device=device,min_face_size=40).to(device)
prcnn.load_state_dict(torch.load('resources/model/PRCNN.pth', map_location=device))
prcnn.eval()
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