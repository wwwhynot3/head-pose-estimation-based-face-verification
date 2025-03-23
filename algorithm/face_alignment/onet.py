import torch
from torch import nn
import os


class ONet(nn.Module):
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


class LiteONet(nn.Module):
    """优化后的关键点检测网络，仅保留ONet的特征提取和关键点预测分支"""

    def __init__(self, pretrained=True, device='cpu'):
        super().__init__()

        # 特征提取部分（与原始ONet完全一致）
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

        # 全连接层调整
        self.dense5 = nn.Linear(1152, 256)  # 保持原始维度
        self.prelu5 = nn.PReLU(256)

        # 关键点输出层（保持原始名称以便加载权重）
        self.dense6_3 = nn.Linear(256, 10)  # 原始关键点分支

        # 加载预训练权重
        if pretrained:
            self._load_pretrained_weights()
        if device is not None:
            self.to(device)
            self.eval()

    def forward(self, x):
        # 特征提取流程
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)

        x = self.prelu2(self.conv2(x))
        x = self.pool2(x)

        x = self.prelu3(self.conv3(x))
        x = self.pool3(x)

        x = self.prelu4(self.conv4(x))

        # 全连接处理
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense5(x.view(x.size(0), -1))
        x = self.prelu5(x)

        # 关键点输出 [batch, 10] -> [batch, 5, 2]
        landmarks = self.dense6_3(x).view(-1, 5, 2)
        return landmarks

    def _load_pretrained_weights(self):
        """加载原始ONet的预训练权重（自动跳过不存在的层）"""
        original_net = ONet(pretrained=True)
        state_dict = {
            k: v for k, v in original_net.state_dict().items()
            if not k.startswith(('dense6_1', 'dense6_2', 'softmax6_1'))
        }

        # 权重名称映射（确保关键点层正确加载）
        name_mapping = {
            'dense6_3.weight': 'dense6_3.weight',
            'dense6_3.bias': 'dense6_3.bias'
        }

        # 重映射关键点层名称
        adjusted_state_dict = {}
        for k, v in state_dict.items():
            if k in name_mapping:
                new_k = name_mapping[k]
                adjusted_state_dict[new_k] = v
            else:
                adjusted_state_dict[k] = v
        self.load_state_dict(adjusted_state_dict, strict=False)
        print("成功加载预训练权重，已跳过无关层")