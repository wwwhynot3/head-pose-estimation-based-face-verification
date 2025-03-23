import torch
import numpy as np
from torch import nn
import os


class ONet(nn.Module):
    """MTCNN ONet.

    Keyword Arguments:
        pretrained {bool} -- Whether or not to load saved pretrained weights (default: {True})
    """

    def __init__(self, pretrained=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        self.prelu4 = nn.PReLU(128)
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)
        self.dense6_1 = nn.Linear(256, 2)
        self.softmax6_1 = nn.Softmax(dim=1)
        self.dense6_2 = nn.Linear(256, 4)
        self.dense6_3 = nn.Linear(256, 10)

        self.training = False

        if pretrained:
            state_dict_path = os.path.join(os.path.dirname(__file__), '../../resources/model/onet.pt')
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense5(x.view(x.shape[0], -1))
        x = self.prelu5(x)
        a = self.dense6_1(x)
        a = self.softmax6_1(a)
        b = self.dense6_2(x)
        c = self.dense6_3(x)
        return b, c, a

class KeypointNet(ONet):
    """人脸关键点检测网络（基于O-Net改进）

    输入：已裁剪的人脸区域图像（建议48x48像素）
    输出：5个人脸关键点坐标（左眼、右眼、鼻尖、左嘴角、右嘴角）
    """

    def __init__(self, pretrained=True):
        super().__init__(pretrained=False)  # 不加载原始全连接层参数

        # 保留原始卷积层结构
        # 修改全连接层：仅保留关键点预测分支
        self.dense6_3 = nn.Linear(256, 10)  # 5个关键点 x 2个坐标

        # 加载预训练参数（跳过分类和回归层）
        if pretrained:
            original_weights = torch.load(os.path.join(os.path.dirname(__file__), '../../resources/model/onet.pt'))

            # 筛选需要的参数（排除分类和回归分支）
            filtered_weights = {k: v for k, v in original_weights.items()
                                if not k.startswith('dense6_1')
                                and not k.startswith('dense6_2')}

            # 加载权重（严格匹配）
            self.load_state_dict(filtered_weights, strict=False)

    def forward(self, x):
        # 前向传播流程（保留特征提取部分）
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense5(x.view(x.shape[0], -1))
        x = self.prelu5(x)

        # 仅返回关键点坐标（原始输出为10维）
        keypoints = self.dense6_3(x)
        return keypoints.view(-1, 5, 2)  # 转换为(B, 5, 2)形状

