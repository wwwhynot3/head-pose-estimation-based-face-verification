import torch
from torch.nn.functional import interpolate
from torchvision.transforms import functional as F
from torchvision.ops.boxes import batched_nms
from PIL import Image
import numpy as np
import os
import math

from algorithm.face_alignment.onet import KeypointNet

# OpenCV is optional, but required if using numpy arrays instead of PIL
try:
    import cv2
except:
    pass
# ---------------------- 关键点检测主函数 ----------------------
def detect_face(imgs, keypoint_net, device):
    """人脸关键点检测函数（修正坐标解码版）"""

    # 输入统一处理
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]

    # 预处理参数
    INPUT_SIZE = 48
    NORM_MEAN = 127.5
    NORM_SCALE = 0.0078125

    processed = []
    original_sizes = []

    for img in imgs:
        # 转换为Tensor并保持数值范围
        if isinstance(img, Image.Image):
            img_tensor = F.to_tensor(img) * 255.0
        elif isinstance(img, np.ndarray):
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        elif isinstance(img, torch.Tensor):
            img_tensor = img.clone().detach().float()
            if img_tensor.max() <= 1.0:
                img_tensor *= 255.0
        else:
            raise TypeError(f"Unsupported type: {type(img)}")

        # 记录原始尺寸 (H, W)
        h, w = img_tensor.shape[1], img_tensor.shape[2]
        original_sizes.append((h, w))

        # 调整尺寸并标准化
        resized = interpolate(img_tensor.unsqueeze(0), size=INPUT_SIZE, mode="bilinear")
        normalized = (resized - NORM_MEAN) * NORM_SCALE
        processed.append(normalized)

    # 合并批次
    batch = torch.cat(processed).to(device)

    # 关键点预测
    with torch.no_grad():
        raw_kpts = keypoint_net(batch)  # 形状：[N,5,2]

    # 坐标解码（重要修正）
    final_keypoints = []
    for i, (h, w) in enumerate(original_sizes):
        # 获取原始输出（注意：这里假设网络直接输出图像尺度坐标）
        kpts = raw_kpts[i].cpu().numpy()

        # 动态缩放（根据实际网络设计调整以下参数）
        scale_factor = 48.0 / INPUT_SIZE  # 假设网络输出基于48px的坐标
        kpts *= scale_factor

        # 缩放到原始图像尺寸
        kpts[:, 0] *= w / 48.0  # X坐标缩放
        kpts[:, 1] *= h / 48.0  # Y坐标缩放

        final_keypoints.append(kpts)

    return np.array(final_keypoints)


# 验证代码
"""
# 测试样本（48x48的虚拟数据）
test_input = torch.rand(3,48,48).unsqueeze(0) * 255
keypoint_net.eval()

# 运行检测
keypoints = detect_face(test_input, keypoint_net, device)

# 预期输出应分布在图像范围内
print("关键点坐标范围:")
print(f"X: {keypoints[0,:,0].min()} - {keypoints[0,:,0].max()}")
print(f"Y: {keypoints[0,:,1].min()} - {keypoints[0,:,1].max()}")
"""

# ---------------------- 使用示例 ----------------------
if __name__ == "__main__":
    # 初始化模型
    device = 'cpu'
    kp_model = KeypointNet(pretrained=True).to(device)
    kp_model.eval()

    # 加载测试图像
    test_images = [
        Image.open('resources/pictures/input/1-1.jpeg'),
    ]

    # 检测关键点
    keypoints = detect_face_keypoints(test_images, kp_model, device)

    # 打印结果
    for i, kpts in enumerate(keypoints):
        print(f"人脸{i + 1}关键点：")
        print(f"左眼: ({kpts[0][0]:.1f}, {kpts[0][1]:.1f})")
        print(f"右眼: ({kpts[1][0]:.1f}, {kpts[1][1]:.1f})")
        print(f"鼻子: ({kpts[2][0]:.1f}, {kpts[2][1]:.1f})")
        print(f"左嘴角: ({kpts[3][0]:.1f}, {kpts[3][1]:.1f})")
        print(f"右嘴角: ({kpts[4][0]:.1f}, {kpts[4][1]:.1f})")
        print()
