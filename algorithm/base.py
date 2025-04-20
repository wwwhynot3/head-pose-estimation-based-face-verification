import torch
from torchvision import transforms as trans
import torchvision

from algorithm import prepare_facebank
from algorithm.model import MobileFaceNet, PRCNN, HopeNet, ShuffledHopeNet
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
mobilefacenet.eval()

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