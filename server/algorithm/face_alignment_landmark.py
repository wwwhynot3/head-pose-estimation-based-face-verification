import cv2
import numpy as np
from numpy.linalg import inv, lstsq
from algorithm.face_detection import detect_face

# 参考面部关键点（左眼、右眼、鼻尖、左嘴角、右嘴角）
REFERENCE_FACIAL_POINTS = np.array([
    [30.29459953, 51.69630051],  # 左眼中心
    [65.53179932, 51.50139999],  # 右眼中心
    [48.02519989, 71.73660278],  # 鼻尖
    [33.54930115, 92.3655014],  # 左嘴角
    [62.72990036, 92.20410156]  # 右嘴角
], dtype=np.float32)


def findNonreflectiveSimilarity(uv, xy, K=2):
    """计算非反射相似变换矩阵"""
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))
    y = xy[:, 1].reshape((-1, 1))

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))

    u = uv[:, 0].reshape((-1, 1))
    v = uv[:, 1].reshape((-1, 1))
    U = np.vstack((u, v))

    if np.linalg.matrix_rank(X) >= 2 * K:
        r, _, _, _ = lstsq(X, U, rcond=None)
        r = np.squeeze(r)
    else:
        raise Exception('cp2tform:twoUniquePointsReq')

    sc, ss, tx, ty = r
    Tinv = np.array([
        [sc, -ss, 0],
        [ss, sc, 0],
        [tx, ty, 1]
    ])

    T = inv(Tinv)
    T = T[:2, :]  # 提取2x3变换矩阵
    return T

def align_face(faces, landmarks):
    aligned_faces = []
    for face, landmark in zip(faces, landmarks):
        detected_landmarks = lm.astype(np.float32).reshape(-1, 2)  # 确保形状为(5,2)

        # 计算相似变换矩阵
        similar_trans_matrix = findNonreflectiveSimilarity(detected_landmarks, REFERENCE_FACIAL_POINTS)

        # 应用仿射变换进行人脸对齐
        aligned_face = cv2.warpAffine(
            face,
            similar_trans_matrix,
            (112, 112),
            borderMode=cv2.BORDER_REPLICATE
        )
        aligned_faces.append(aligned_face)
    return aligned_faces

if __name__ == "__main__":
    frame = cv2.imread('resources/pictures/input/1.jpeg')

    # 检测人脸并获取关键点（确保landmark=True）
    boxes, faces, probss, landmarks = detect_face(frame, min_prob=0.9, landmark=True)

    print(f"检测到 {len(boxes)} 张人脸")

    for i, (box, face, lm) in enumerate(zip(boxes, faces, landmarks)):
        if lm is None or lm.size == 0:
            continue

        # 关键点顺序调整（如果MTCNN输出顺序与参考点不一致需调整）
        # 假设MTCNN输出顺序为：左眼、右眼、鼻尖、左嘴角、右嘴角（与参考点一致）
        detected_landmarks = lm.astype(np.float32).reshape(-1, 2)  # 确保形状为(5,2)

        # 计算相似变换矩阵
        similar_trans_matrix = findNonreflectiveSimilarity(detected_landmarks, REFERENCE_FACIAL_POINTS)

        # 应用仿射变换进行人脸对齐
        aligned_face = cv2.warpAffine(
            face,
            similar_trans_matrix,
            (112, 112),
            borderMode=cv2.BORDER_REPLICATE
        )

        # 保存对齐后的人脸（注意颜色空间转换）
        path = f'resources/pictures/output/{i}_out.jpeg'
        cv2.imwrite(path, cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB))