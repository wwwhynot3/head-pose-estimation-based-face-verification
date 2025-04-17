import cv2
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
from algorithm.base import hopenet, shuffledhopenet, hopenet_transform, device
import algorithm.face_pose_estimation.utils

def _testt_hopenet(model, pic):
    # 4. 推理预测
    with torch.no_grad():
        images = pic.to(device)  # 替换Variable(input_img.cuda())
        yaw, pitch, roll = model(images)

    # 5. 转换欧拉角[1](@ref)
    # idx_tensor = torch.FloatTensor([i for i in range(66)]).cuda(args.gpu_id)
    idx_tensor = torch.FloatTensor([i for i in range(66)]).to(device)
    yaw_pred = algorithm.face_pose_estimation.utils.utils.softmax_temperature(yaw.data, 1)
    pitch_pred = algorithm.face_pose_estimation.utils.utils.softmax_temperature(pitch.data, 1)
    roll_pred = algorithm.face_pose_estimation.utils.utils.softmax_temperature(roll.data, 1)

    yaw_deg = (torch.sum(yaw_pred * idx_tensor, 1).cpu()[0] * 3 - 99).item()
    pitch_deg = (torch.sum(pitch_pred * idx_tensor, 1).cpu()[0] * 3 - 99).item()
    roll_deg = (torch.sum(roll_pred * idx_tensor, 1).cpu()[0] * 3 - 99).item()
    return yaw_deg, pitch_deg, roll_deg

def face_pose_estimate_single(img):
    input_img = hopenet_transform(img).unsqueeze(0)  # 增加batch维度
    return _testt_hopenet(shuffledhopenet, img)
    
