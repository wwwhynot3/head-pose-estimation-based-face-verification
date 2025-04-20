import torch
import torch.nn.functional as F
from algorithm.base import shuffledhopenet, hopenet_transform, device

# 全局配置
# torch.set_num_threads(4)  # 根据CPU核心数调整线程数

# 初始化模型和预计算张量
# hopenet_model = shuffledhopenet.to(device).eval()
idx_tensor = torch.arange(66, dtype=torch.float32, device=device)


def _softmax_temperature(tensor, temperature):
    """优化后的温度调节softmax"""
    scaled = tensor / temperature
    return F.softmax(scaled - scaled.max(dim=1, keepdim=True).values, dim=1)


def _batch_pose_predict(model, batch_tensor):
    """批量姿态估计核心函数"""
    with torch.no_grad():
        yaw, pitch, roll = model(batch_tensor)

    # 并行处理所有输出
    yaw_pred = _softmax_temperature(yaw, 1)
    pitch_pred = _softmax_temperature(pitch, 1)
    roll_pred = _softmax_temperature(roll, 1)

    # 向量化计算角度
    yaw_deg = (torch.sum(yaw_pred * idx_tensor, dim=1) * 3 - 99).cpu().numpy()
    pitch_deg = (torch.sum(pitch_pred * idx_tensor, dim=1) * 3 - 99).cpu().numpy()
    roll_deg = (torch.sum(roll_pred * idx_tensor, dim=1) * 3 - 99).cpu().numpy()

    return list(zip(yaw_deg, pitch_deg, roll_deg))


def face_pose_estimate_single(img):
    """单张图像姿态估计"""
    input_tensor = hopenet_transform(img).unsqueeze(0).to(device)
    return _batch_pose_predict(input_tensor)[0]


def face_pose_estimate_batch(model, img_list):
    """批量图像姿态估计"""
    #img从BGR转换为RGB
    # 使用生成器表达式减少内存占用
    # batch = torch.stack([hopenet_transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in img_list]).to(device)
    batch = torch.stack([hopenet_transform(img) for img in img_list]).to(device)

    return _batch_pose_predict(model, batch)


