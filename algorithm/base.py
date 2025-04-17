import torch
from torchvision import transforms as trans
import torchvision
from algorithm.model import MobileFaceNet, PRCNN, HopeNet, ShuffledHopeNet
from algorithm.model.shufflehopenet import ShuffledHopeNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
facebank_path='resources/face_lib'
faces_dir='resources/faces'
output_dir='resources/results'
mobilefacenet_path = 'resources/model/MobileFace_Net'
pnet_path = 'resources/model/pnet.pt'
rnet_path = 'resources/model/rnet.pt'
hopenet_path = 'resources/model/hopenet.pkl'
shuffledhopenet_path = 'resources/model/shuffledhopenet.pkl'

# 加载模型
mobilefacenet = MobileFaceNet(512).to(device)
mobilefacenet.load_state_dict(torch.load(mobilefacenet_path, map_location=device))
mobilefacenet.eval()

prcnn = PRCNN(image_size=160, thresholds=[0.8, 0.9],min_face_size=40,pnet_path=pnet_path, rnet_path=rnet_path, device=device).to(device)
# prcnn.load_state_dict(torch.load(prcnn_path, map_location=device))
prcnn.eval()

hopenet = HopeNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
hopenet.load_state_dict(torch.load(hopenet_path, map_location=device))
# hopenet = hopenet.to(device)  # 替换model.cuda()
hopenet.eval()

shuffledhopenet = ShuffledHopeNet([4, 8, 4], [24, 116, 232, 464, 1024])
shuffledhopenet.load_state_dict(torch.load(shuffledhopenet_path, map_location=device), strict=False)
shuffledhopenet.eval()

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