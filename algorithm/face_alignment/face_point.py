import numpy as np
import cv2
from .onet import LiteONet
from torchvision.transforms import functional as F
import torch
def get_landmarks(
        cropped_face: np.ndarray,  # 已裁剪的人脸区域 [H,W,C]
        model: LiteONet,
        device: str = 'cpu'
) -> np.ndarray:
    """
    在已裁剪的人脸图像上获取绝对坐标关键点

    参数:
        cropped_face: 裁剪后的人脸区域，任意尺寸
        model: 已加载的LiteONet模型
        device: 计算设备

    返回:
        关键点绝对坐标数组 [5,2]，相对于输入图像坐标系
    """
    # 记录原始尺寸
    orig_h, orig_w = cropped_face.shape[:2]

    # 预处理
    face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    input_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float()
    input_tensor = F.resize(input_tensor, [48, 48])
    input_tensor = (input_tensor - 127.5) * 0.0078125  # 关键归一化

    # 模型推理
    with torch.no_grad():
        model.eval()
        landmarks = model(input_tensor.unsqueeze(0).to(device))
        landmarks = landmarks.cpu().numpy()[0]

    # 绝对坐标转换
    scale_x = orig_w / 48  # 注意：分母是模型输入尺寸
    scale_y = orig_h / 48
    abs_landmarks = landmarks * np.array([scale_x, scale_y])
    print(abs_landmarks)
    return abs_landmarks.astype(int)
