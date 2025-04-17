import os
import cv2
import numpy as np
from pathlib import Path
from algorithm.base import *
from algorithm.model import MobileFaceNet, l2_norm

def __prepare_facebank():
    embeddings = []
    name_list = ['Unknown']

    for person_dir in Path(facebank_path).iterdir():
        if not person_dir.is_dir() or person_dir.name.startswith('.'):
            continue

        embs = []
        for img_file in person_dir.glob('*.*'):
            if img_file.suffix.lower() not in ['.jpg', '.png', '.jpeg']:
                continue

            # 读取并预处理图像（仅保留transform的resize）
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 确保颜色通道正确

            # 特征提取与增强
            with torch.no_grad():
                # 原始图像
                img_tensor = mobilefacenet_transform(img).unsqueeze(0).to(device)
                emb = mobilefacenet(img_tensor)

                # 镜像增强
                mirror_img = cv2.flip(img, 1)
                mirror_tensor = mobilefacenet_transform(mirror_img).unsqueeze(0).to(device)
                emb_mirror = mobilefacenet(mirror_tensor)

                # 融合特征并归一化
                fused_emb = l2_norm(emb + emb_mirror)
                embs.append(fused_emb)

        if embs:
            # 聚合特征并二次归一化（关键修正点）
            avg_emb = torch.cat(embs).mean(dim=0, keepdim=True)
            avg_emb = l2_norm(avg_emb)  # 必须再次归一化
            embeddings.append(avg_emb)
            name_list.append(person_dir.name)

    return torch.cat(embeddings) if embeddings else torch.Tensor(), np.array(name_list)



def face_recognition_pipeline(
        threshold=0.6,  # 调整默认阈值到更合理的范围
):
    # device = torch.device(device)
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    # model = MobileFaceNet(512).to(device)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model.eval()

    # 图像预处理（保持与训练时一致）
    # transform = trans.Compose([
    #     trans.ToPILImage(),
    #     trans.Resize((112, 112)),  # 单次resize即可
    #     trans.ToTensor(),
    #     trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # ])



    # 加载/构建特征库
    facebank_file = Path(facebank_path) / 'facebank.pth'
    names_file = Path(facebank_path) / 'names.npy'

    if facebank_file.exists() and names_file.exists():
        targets = torch.load(facebank_file, map_location=device)
        names = np.load(names_file)
    else:
        targets, names = __prepare_facebank()
        torch.save(targets, facebank_file)
        np.save(names_file, names)

    # 处理待识别图片
    processed_count = 0
    for img_file in Path(faces_dir).glob('*.*'):
        if img_file.suffix.lower() not in ['.jpg', '.png', '.jpeg']:
            continue

        # 读取图像（仅做一次resize）
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 特征提取
        with torch.no_grad():
            # 原始图像
            img_tensor = mobilefacenet_transform(img).unsqueeze(0).to(device)
            emb = mobilefacenet(img_tensor)

            # 镜像增强
            mirror_img = cv2.flip(img, 1)
            mirror_tensor = mobilefacenet_transform(mirror_img).unsqueeze(0).to(device)
            emb_mirror = mobilefacenet(mirror_tensor)

            # 融合特征并归一化
            fused_emb = l2_norm(emb + emb_mirror)

        # 验证特征归一化
        print(f"特征模长: {torch.norm(fused_emb).item():.4f}")  # 应该严格等于1.0

        # 计算相似度（使用余弦相似度更合理）
        if len(targets) > 0:
            cosine_sim = torch.mm(fused_emb, targets.T).squeeze()  # [1, n] -> [n]
            max_sim, max_idx = torch.max(cosine_sim, dim=0)
            max_sim = max_sim.item()
            max_idx = max_idx.item()
        else:
            max_sim = -1
            max_idx = -1

        # 确定身份
        name = "Unknown"
        if max_sim >= threshold and max_idx != -1:
            name = names[max_idx + 1]  # +1 跳过Unknown

        # 可视化与保存
        display_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(display_img, f"{name} ({max_sim:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imwrite(str(Path(output_dir) / img_file.name), display_img)
        processed_count += 1
        print(f"Processed: {img_file.name} -> {name} (score: {max_sim:.2f})")

    return f"Completed! Processed {processed_count} images."


# 示例调用
if __name__ == "__main__":
    print(face_recognition_pipeline(
        threshold=0.6  # 使用余弦相似度时阈值设为0.6左右
    ))

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = torch.Tensor(2, 3, 112, 112).to(device)
    net = MobileFaceNet(512).to(device)
    x = net(input)
    print(x.shape)

