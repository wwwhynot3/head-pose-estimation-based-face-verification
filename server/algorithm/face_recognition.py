import os
import cv2
import numpy as np
from pathlib import Path
import torch
from torchvision import transforms
from algorithm.base import *
from algorithm.model import MobileFaceNet
from algorithm.model.mobilefacenet import l2_norm


def _extract_embeddings_batch(images, model):
    """
    批量提取图像特征的统一方法
    """
    if len(images) == 0:
        return []
    img_tensors = torch.stack([mobilefacenet_transform(img) for img in images]).to(device)

    with torch.no_grad():
        # 处理原始图像
        orig_embs = model(img_tensors)

        # 处理镜像图像
        mirror_imgs = [cv2.flip(img, 1) for img in images]
        mirror_tensors = torch.stack([mobilefacenet_transform(img) for img in mirror_imgs]).to(device)
        mirror_embs = model(mirror_tensors)

        # 融合特征并归一化
        fused_embs = l2_norm(orig_embs + mirror_embs)

    return fused_embs
        # return orig_embs

def face_recognition_batch(image_batch, model, threshold=0.6):
    """
    批量识别人脸的核心方法
    """
    if len(image_batch) == 0:
        return [], []
    # 批量提取特征
    query_embs = _extract_embeddings_batch(image_batch, model)

    # 计算相似度
    if len(face_targets) > 0:
        cosine_sim = torch.mm(query_embs, face_targets.T)  # [batch_size, num_targets]
        max_sims, max_indices = torch.max(cosine_sim, dim=1)
    else:
        max_sims = torch.full((len(image_batch),), -1.0, device=device)
        max_indices = torch.full((len(image_batch),), -1, device=device)

    # 转换为CPU数据
    max_sims = max_sims.cpu().numpy()
    max_indices = max_indices.cpu().numpy()

    # 生成识别结果
    results = []
    for sim, idx in zip(max_sims, max_indices):
        if sim >= threshold and idx != -1:
            results.append(face_names[idx + 1])  # +1 跳过Unknown
        else:
            results.append("Unknown")

    return results, max_sims


# def face_recognition_batch(faces, threshold=0.4, model=mobilefacenet):
#     """
#     处理整个目录的入口方法
#     """
#     # os.makedirs(output_dir, exist_ok=True)
#     image_nos = range(len(faces))
#     valid_images = faces
#
#     # 批量识别
#     results, scores = _face_recognition_batch(valid_images, model, threshold)
#
#     # # 保存结果
#     # for img_file, img, name, score in zip(image_nos, valid_images, results, scores):
#     #     print(f"Processing Face {img_file}: {name} ({score:.2f})")
#     #     display_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     #     cv2.putText(display_img, f"{name} ({score:.2f})", (10, 30),
#     #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#     #     cv2.imwrite(str(Path(output_dir) / img_file.name), display_img)
#     #
#     # return f"Processed {len(valid_images)} images."
#     return zip(image_nos, results, scores)

def process_directory(model=mobilefacenet, threshold=0.4):
    """
    处理整个目录的入口方法
    """
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []
    valid_images = []

    # 收集有效图像
    for img_file in Path(faces_dir).glob('*.*'):
        if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_paths.append(img_file)
            valid_images.append(img)

    # 批量识别
    results, scores = face_recognition_batch(valid_images, model, threshold)

    # 保存结果
    for img_file, img, name, score in zip(image_paths, valid_images, results, scores):
        print(f"Processing {img_file.name}: {name} ({score:.2f})")
        display_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(display_img, f"{name} ({score:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imwrite(str(Path(output_dir) / img_file.name), display_img)

    return f"Processed {len(valid_images)} images."


# 使用示例
if __name__ == "__main__":
    # 初始化模型
    mobilefacenet = MobileFaceNet().to(device).eval()

    # 准备特征库
    facebank_path = "./facebank"
    face_targets, face_names = prepare_facebank(facebank_path, mobilefacenet)

    # 处理待识别目录
    input_dir = "./faces"
    output_dir = "./results"
    process_directory(input_dir, output_dir, mobilefacenet, face_targets, face_names)