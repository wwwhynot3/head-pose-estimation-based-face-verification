import torch
import numpy as np
from facenet_pytorch import MTCNN
from facenet_pytorch.models.utils.detect_face import nms_numpy, pad


class PNetOnlyDetector:
    def __init__(self, min_face_size=20, thresholds=0.6, device='cuda'):
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.device = device if torch.cuda.is_available() else 'cpu'

        # 加载预训练的 P-Net
        self.mtcnn = MTCNN(device=self.device)
        self.pnet = self.mtcnn.pnet
        self.pnet.eval()

    def _generate_scales(self, im):
        """生成图像金字塔的缩放因子"""
        height, width = im.shape[1], im.shape[2]
        scales = []
        factor = 0.707  # 缩放因子（与原始 MTCNN 一致）
        min_size = min(height, width)

        # 计算缩放链
        m = 12.0 / self.min_face_size  # 初始缩放比例
        min_size *= m
        while min_size >= 12:
            scales.append(m)
            m *= factor
            min_size *= factor
        return scales

    def detect(self, im):
        """核心检测逻辑"""
        if not isinstance(im, torch.Tensor):
            im = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        # 生成图像金字塔的缩放因子
        scales = self._generate_scales(im[0].cpu().numpy())

        all_boxes = []
        for scale in scales:
            # 1. 缩放图像
            scaled_im = torch.nn.functional.interpolate(
                im,
                scale_factor=scale,
                mode='bilinear',
                align_corners=False
            )

            # 2. 运行 P-Net
            with torch.no_grad():
                output = self.pnet(scaled_im)
                probs = output[1].cpu().detach().numpy()[0, 1, :, :]
                offsets = output[0].cpu().detach().numpy()

            # 3. 提取候选框（阈值过滤）
            indices = np.where(probs > self.thresholds)
            if indices[0].size == 0:
                continue

            # 4. 将候选框映射回原始坐标
            boxes = self._generate_boxes(
                probs, offsets, scale,
                im.shape[3], im.shape[2]  # 原始宽高
            )
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return np.array([]), np.array([])

        # 合并所有候选框
        all_boxes = np.vstack(all_boxes)

        # 5. nms_numpy 去重
        keep = nms_numpy(all_boxes[:, :4], all_boxes[:, 4], threshold=0.7)
        final_boxes = all_boxes[keep]

        # 6. 最终置信度过滤
        valid = final_boxes[:, 4] > self.thresholds
        final_boxes = final_boxes[valid]

        return final_boxes[:, :4].astype(int), final_boxes[:, 4]

    def _generate_boxes(self, probs, offsets, scale, orig_w, orig_h):
        """将 P-Net 输出转换为候选框坐标"""
        stride = 2
        cell_size = 12

        # 获取候选框的坐标偏移
        inds = np.where(probs > self.thresholds)
        if inds[0].size == 0:
            return np.array([])

        # 计算候选框的中心点（缩放后坐标系）
        dx1, dy1, dx2, dy2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
        offsets = np.array([dx1, dy1, dx2, dy2])
        scores = probs[inds[0], inds[1]]

        # 计算候选框的原始坐标（缩放后图像）
        x1 = (stride * inds[1] + 1) / scale
        y1 = (stride * inds[0] + 1) / scale
        x2 = (stride * inds[1] + 1 + cell_size) / scale - 1
        y2 = (stride * inds[0] + 1 + cell_size) / scale - 1

        # 应用偏移量
        boxes = np.vstack([
            x1 - dx1 * (x2 - x1),
            y1 - dy1 * (y2 - y1),
            x2 + dx2 * (x2 - x1),
            y2 + dy2 * (y2 - y1),
            scores
        ]).T

        # 裁剪到图像边界
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)

        return boxes


# 测试示例
if __name__ == "__main__":
    import cv2

    # 初始化检测器
    detector = PNetOnlyDetector(min_face_size=20, thresholds=0.6)

    # 读取测试图像
    img = cv2.imread("test.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 检测人脸
    boxes, scores = detector.detect(img)

    # 绘制结果
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite("result.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("检测到的人脸框坐标:", boxes)