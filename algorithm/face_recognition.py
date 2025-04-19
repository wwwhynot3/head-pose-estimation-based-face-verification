import os
import cv2
import numpy as np
from pathlib import Path
import torch
from algorithm.base import mobilefacenet_transform, mobilefacenet, facebank_path, device, output_dir, faces_dir
from algorithm.model import MobileFaceNet, l2_norm


class FaceBank:
    def __init__(self, facebank_path, device='cuda'):
        self.facebank_path = Path(facebank_path)
        self.device = device
        self.targets = None
        self.names = ['Unknown']
        self.model = mobilefacenet

    def build(self):
        """构建特征库并保存"""
        embeddings = []
        name_list = ['Unknown']

        # 遍历人物目录
        for person_dir in self._valid_dirs():
            embs = self._process_person(person_dir)
            if embs:
                avg_emb = self._aggregate_embeddings(embs)
                embeddings.append(avg_emb)
                name_list.append(person_dir.name)

        # 保存特征库
        self._save_resources(
            torch.stack(embeddings) if embeddings else torch.Tensor(),
            np.array(name_list)
        )

    def load(self):
        """加载预生成的特征库"""
        targets = torch.load(self.facebank_path / 'facebank.pth', map_location=self.device)
        names = np.load(self.facebank_path / 'names.npy', allow_pickle=True)
        return targets, names

    def _valid_dirs(self):
        """验证有效的人物目录"""
        return [d for d in self.facebank_path.iterdir()
                if d.is_dir() and not d.name.startswith('.')]

    def _process_person(self, person_dir):
        """处理单个人物的所有图像"""
        embs = []
        for img_file in person_dir.glob('*.*'):
            if img_file.suffix.lower() not in ['.jpg', '.png', '.jpeg']:
                continue

            img = self._load_image(img_file)
            if img is None:
                continue

            with torch.no_grad():
                # 原始+镜像特征融合
                fused_emb = self._get_fused_embedding(img)
                embs.append(fused_emb)
        return embs

    def _get_fused_embedding(self, img):
        """获取融合镜像增强后的特征"""
        # 原始图像
        img_tensor = mobilefacenet_transform(img).unsqueeze(0).to(self.device)
        emb = self.model(img_tensor)

        # 镜像增强
        mirror_img = cv2.flip(img, 1)
        mirror_tensor = mobilefacenet_transform(mirror_img).unsqueeze(0).to(self.device)
        emb_mirror = self.model(mirror_tensor)

        return l2_norm(emb + emb_mirror)

    def _aggregate_embeddings(self, embs):
        """聚合单人多张图像的特征"""
        stacked = torch.cat(embs)
        avg_emb = torch.mean(stacked, dim=0, keepdim=True)
        return l2_norm(avg_emb)

    def _save_resources(self, targets, names):
        """保存特征库和名称列表"""
        torch.save(targets, self.facebank_path / 'facebank.pth')
        np.save(self.facebank_path / 'names.npy', names, allow_pickle=True)

    def _load_image(self, img_file):
        """加载并验证图像"""
        img = cv2.imread(str(img_file))
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 初始化特征库
facebank = FaceBank(facebank_path, device)
# 加载/构建特征库
if (facebank.facebank_path / 'facebank.pth').exists():
    targets, names = facebank.load()
else:
    facebank.build()
    targets, names = facebank.load()
# 创建输出目录
output_path = Path(output_dir)
os.makedirs(output_path, exist_ok=True)



def face_recognition_pipeline(
    faces,
    threshold=0.6,
):
    """人脸识别流水线"""

    # 批量处理图像
    processed_count = 0
    for img_file in Path(faces_dir).glob('*.*'):
        if img_file.suffix.lower() not in ['.jpg', '.png', '.jpeg']:
            continue

        # 识别处理
        img = cv2.cvtColor(cv2.imread(str(img_file)), cv2.COLOR_BGR2RGB)
        name, max_sim = _process_single_image(img, facebank.model, threshold)

        # 保存结果
        _save_result(img, name, max_sim, img_file)
        processed_count += 1
        print(f"Processed: {img_file.name} -> {name} (score: {max_sim:.2f})")

    return f"Completed! Processed {processed_count} images."


def _process_single_image(img, model, threshold):
    """处理单张图像"""
    with torch.no_grad():
        # 获取融合特征
        img_tensor = mobilefacenet_transform(img).unsqueeze(0).to(device)
        emb = model(img_tensor)

        mirror_img = cv2.flip(img, 1)
        mirror_tensor = mobilefacenet_transform(mirror_img).unsqueeze(0).to(device)
        emb_mirror = model(mirror_tensor)

        fused_emb = l2_norm(emb + emb_mirror)

    # 计算相似度
    if len(targets) > 0:
        cosine_sim = torch.mm(fused_emb, targets.T).squeeze()
        max_sim, max_idx = torch.max(cosine_sim, dim=0)
        max_sim = max_sim.item()
        max_idx = max_idx.item()
    else:
        max_sim = -1
        max_idx = -1

    # 确定身份
    name = "Unknown"
    if max_sim >= threshold and max_idx != -1:
        name = names[max_idx + 1]  # 跳过Unknown

    return name, max_sim


def _save_result(img, name, similarity, img_file):
    """可视化并保存结果"""
    display_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.putText(display_img,
                f"{name} ({similarity:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)
    cv2.imwrite(str(output_dir / img_file.name), display_img)