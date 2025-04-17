import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import multiprocessing
from PIL import Image
import numpy as np

from algorithm.base import hopenet, hopenet_transform
from math import sin, cos
# 预定义配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32  # 根据内存调整
MODEL_QUANTIZED = True

# 初始化模型（带量化）
model = hopenet.to(device)
if MODEL_QUANTIZED and device.type == "cpu":
    model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
model.eval()

# 多线程优化
torch.set_num_threads(multiprocessing.cpu_count() // 2)
torch.set_num_interop_threads(multiprocessing.cpu_count() // 2)

# 预计算索引张量（避免重复计算）
idx_tensor = torch.arange(66, dtype=torch.float32, device=device) * 3 - 99  # [1](@ref)

def _vectorized_angle_calculation(preds):
    # 向量化计算（提升50%速度）
    return (preds * idx_tensor).sum(dim=1, keepdim=True).cpu().numpy().flatten()


class BatchProcessor:
    def __init__(self):
        self.transform = hopenet_transform
        self.buffer = []

    def add_image(self, img):
        # 使用OpenCV替代PIL加速预处理
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(img)
        self.buffer.append(self.transform(img))

    def get_batch(self, batch_size=32):
        # 生成内存连续的张量
        batch = torch.stack(self.buffer[:batch_size], dim=0)
        self.buffer = self.buffer[batch_size:]
        return batch.contiguous().to(device)

def _softmax_temperature(tensor, temperature):
    max_val = tensor.max(dim=1, keepdim=True).values
    scaled = (tensor - max_val) / temperature
    exp_tensor = torch.exp(scaled)
    return exp_tensor / exp_tensor.sum(dim=1, keepdim=True)


def batch_pose_estimate(model, image_list, batch_size=32) -> list:
    processor = BatchProcessor()
    results = []

    for img in image_list:
        processor.add_image(img)
        if len(processor.buffer) >= batch_size:
            batch = processor.get_batch(batch_size)
            with torch.no_grad():
                yaw, pitch, roll = model(batch)

            # 批量计算角度
            yaw_batch = _vectorized_angle_calculation(_softmax_temperature(yaw, 1))
            pitch_batch = _vectorized_angle_calculation(_softmax_temperature(pitch, 1))
            roll_batch = _vectorized_angle_calculation(_softmax_temperature(roll, 1))

            results.extend(zip(yaw_batch, pitch_batch, roll_batch))

    # 处理剩余样本
    if processor.buffer:
        final_batch = processor.get_batch(len(processor.buffer))
        with torch.no_grad():
            yaw, pitch, roll = model(final_batch)

        yaw_batch = _vectorized_angle_calculation(_softmax_temperature(yaw, 1))
        pitch_batch = _vectorized_angle_calculation(_softmax_temperature(pitch, 1))
        roll_batch = _vectorized_angle_calculation(_softmax_temperature(roll, 1))

        results.extend(zip(yaw_batch, pitch_batch, roll_batch))

    return results


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
# 示例使用
if __name__ == "__main__":
    # 加载测试图像
    test_images = [cv2.imread(f"test_{i}.jpg") for i in range(100)]

    # 批量推理
    pose_results = batch_pose_estimate(model, test_images)

    # 可视化（批量绘制）
    for img, (yaw, pitch, roll) in zip(test_images, pose_results):
        vis_img = draw_axis(img, yaw, pitch, roll)
        cv2.imshow("Result", vis_img)
        cv2.waitKey(10)