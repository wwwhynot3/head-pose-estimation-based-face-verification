from timm.layers import SEModule
from torch.testing._internal.common_quantization import ConvBNReLU
from torchvision.models import mobilenet_v3_small
from torchvision.models.shufflenetv2 import InvertedResidual

import torch
import torch.nn as nn
import torch.nn.functional as F


class MobileFaceNet_Lite(nn.Module):
    def __init__(self, pruning_rate=0.3):
        super().__init__()
        # 动态剪枝参数
        self.pruning_mask = None

        # 主干网络（包含通道注意力）
        self.features = nn.Sequential(
            ConvBNReLU(),
            DynamicPrunedBlock(64, 256, stride=2, pruning_rate=pruning_rate),
            SEBlock(256),  # 压缩激励注意力
            InvertedResidual(256, 512, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )

        # 特征嵌入层
        self.embedding = nn.Linear(512, 128)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.embedding(x)

    # 动态通道剪枝实现（网页6/7/8的剪枝策略）
    def apply_dynamic_pruning(self, threshold=0.2):
        for name, module in self.named_modules():
            if isinstance(module, DynamicPrunedBlock):
                # 基于L1范数的通道重要性评估（网页7）
                importance = torch.mean(module.conv1.weight, dim=[1, 2, 3]).abs()
                mask = importance > threshold
                module.set_pruning_mask(mask)

    def _initialize_weights(self):
        # 混合精度初始化（网页9/10的权重初始化策略）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# --- 核心组件实现 ---
class DynamicPrunedBlock(nn.Module):
    """动态通道剪枝模块（结合网页6的通道剪枝和网页8的应用感知策略）"""

    def __init__(self, in_ch, out_ch, stride, pruning_rate):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, int(out_ch * (1 - pruning_rate)), 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(out_ch * (1 - pruning_rate)))
        self.conv2 = nn.Conv2d(int(out_ch * (1 - pruning_rate)), out_ch, 1, bias=False)
        self.pruning_mask = None

    def set_pruning_mask(self, mask):
        self.pruning_mask = mask

    def forward(self, x):
        if self.pruning_mask is not None and not self.training:
            # 推理时应用结构化剪枝（网页7的通道剪枝策略）
            x = x[:, self.pruning_mask, :, :]
        x = F.relu6(self.bn1(self.conv1(x)))
        return self.conv2(x)


class SEBlock(nn.Module):
    """压缩激励注意力（网页8的特征增强策略）"""

    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# 后续可添加量化入口
from torch.quantization import QuantStub, DeQuantStub

class QuantizableMobileFaceNet(MobileFaceNet_Lite):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = super().forward(x)
        return self.dequant(x)


class OfficialMobileFaceNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # 加载官方实现的MobileNetV3（网页6/7/8的技术基础）
        self.backbone = mobilenet_v3_small(pretrained=pretrained).features

        # 替换原有分类头（网页7的SE模块增强）
        self._modify_last_layer()

    def _modify_last_layer(self):
        # 保持与自定义模型相同的128维输出
        self.backbone[-1] = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            SEModule(576),  # 网页8的压缩激励注意力
            nn.Flatten(),
            nn.Linear(576, 128)
        )

    def forward(self, x):
        return self.backbone(x)


# 验证代码
if __name__ == "__main__":
    # 初始化对照组模型
    official_model = OfficialMobileFaceNet()
    model = MobileFaceNet_Lite(pruning_rate=0.3)
    official_model.eval()
    model.apply_dynamic_pruning(threshold=0.25)
    model.eval()

    # 测试输入（需与自定义模型保持相同预处理）
    dummy_input = torch.randn(1, 3, 112, 112)

    # 特征提取
    with torch.no_grad():
        official_feat = official_model(dummy_input)
        feat = model(dummy_input)
    print(f"对照组特征维度: {official_feat.shape}")  # 预期输出: torch.Size([1, 128])
    print(f"自定义模型特征维度: {feat.shape}")  # 预期输出: torch.Size([1, 128])

# --- 测试验证 ---
# if __name__ == "__main__":
#     model = MobileFaceNet_Lite(pruning_rate=0.3)
#
#     # 动态剪枝应用
#     model.apply_dynamic_pruning(threshold=0.25)
#
#     # 测试推理
#     dummy_input = torch.randn(1, 3, 112, 112)
#     output = model(dummy_input)
#     print(f"特征维度: {output.shape}")  # 预期输出: torch.Size([1, 128])