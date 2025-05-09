import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
from algorithm.base import hopenet, shuffledhopenet, hopenet_transform, device
import numpy as np
from math import cos, sin
import torch.nn.functional as F
import cv2


def _softmax_temperature(tensor, temperature):
    # 数值稳定版实现
    scaled = tensor / temperature
    return F.softmax(scaled - scaled.max(dim=1, keepdim=True).values, dim=1)


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img

def _testt_hopenet(model, pic):
    # # 确保模型在推理模式
    # model.eval()

    # 4. 推理预测(移除Variable调用，适配新版PyTorch)
    with torch.no_grad():
        images = pic.to(device)
        # images = Variable(pic)
        yaw, pitch, roll = model(images)

    # 5. 转换欧拉角(直接创建tensor到目标设备)
    idx_tensor = torch.arange(66, dtype=torch.float32, device=device)  # 更高效的创建方式

    # 使用显式dim参数确保高版本兼容性
    yaw_pred = _softmax_temperature(yaw, 1)
    pitch_pred = _softmax_temperature(pitch, 1)
    roll_pred = _softmax_temperature(roll, 1)

    # 计算角度(简化设备转移逻辑)
    yaw_deg = (torch.sum(yaw_pred * idx_tensor, dim=1).cpu().item() * 3 - 99)
    pitch_deg = (torch.sum(pitch_pred * idx_tensor, dim=1).cpu().item() * 3 - 99)
    roll_deg = (torch.sum(roll_pred * idx_tensor, dim=1).cpu().item() * 3 - 99)

    return yaw_deg, pitch_deg, roll_deg


def face_pose_estimate_single(model, img):
    # 确保模型在目标设备
    # model = model.to(device)


    # 预处理并添加batch维度
    input_img = hopenet_transform(img).unsqueeze(0).to(device)

    return _testt_hopenet(model, input_img)