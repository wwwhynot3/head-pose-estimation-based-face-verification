import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ProcessPoolExecutor


def _euler_to_rotation_matrix_batch(pitches, yaws, rolls):
    """批量生成旋转矩阵 (extrinsic XYZ顺序)"""
    angles = np.radians(np.column_stack((pitches, yaws, rolls)))
    rotations = R.from_euler('XYZ', angles, degrees=False)
    return rotations.as_matrix()


def align_face(image, R_matrix, f=1000):
    """优化后的单张图像对齐函数"""
    h, w = image.shape[:2]

    # 定义三维坐标点（以图像中心为原点）
    src_3d = np.array([
        [-w / 2, -h / 2, 0],
        [w / 2, -h / 2, 0],
        [w / 2, h / 2, 0],
        [-w / 2, h / 2, 0]
    ], dtype=np.float32)

    # 应用旋转并投影
    rotated_3d = src_3d @ R_matrix.T
    Z = rotated_3d[:, 2]
    scale = f / (Z + f + 1e-6)
    X = rotated_3d[:, 0] * scale + w / 2
    Y = rotated_3d[:, 1] * scale + h / 2
    dst_2d = np.column_stack((X, Y))

    # 计算透视变换
    src_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M, _ = cv2.findHomography(src_points, dst_2d.astype(np.float32))

    return cv2.warpPerspective(image, M, (w, h))


def align_faces_batch(images, poses):
    """向量化优化后的批量处理"""
    if len(images) != len(poses):
        raise ValueError("图像和姿态列表长度不匹配")
    if len(images) == 0 or len(poses) == 0:
        return []

    pitches, yaws, rolls = zip(*[(p, y, -r) for p, y, r in poses])
    R_matrices = _euler_to_rotation_matrix_batch(pitches, yaws, rolls)
    return [align_face(img, R) for img, R in zip(images, R_matrices)]


def align_faces_batch_parallel(images, poses, workers=4):
    """多进程并行批量处理"""
    pitches, yaws, rolls = zip(*[(p, y, -r) for p, y, r in poses])
    R_matrices = _euler_to_rotation_matrix_batch(pitches, yaws, rolls)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        args = [(img, R) for img, R in zip(images, R_matrices)]
        aligned = list(executor.map(lambda x: align_face(*x), args))
    return aligned

# import cv2
# import numpy as np
#
# def _euler_to_rotation_matrix(pitch, yaw, roll):
#     """将欧拉角转换为3x3旋转矩阵"""
#     # 转换为弧度
#     pitch = np.radians(pitch)
#     yaw = np.radians(yaw)
#     roll = np.radians(roll)
#
#     # 绕X轴旋转（pitch）
#     Rx = np.array([
#         [1, 0, 0],
#         [0, np.cos(pitch), -np.sin(pitch)],
#         [0, np.sin(pitch), np.cos(pitch)]
#     ])
#
#     # 绕Y轴旋转（yaw）
#     Ry = np.array([
#         [np.cos(yaw), 0, np.sin(yaw)],
#         [0, 1, 0],
#         [-np.sin(yaw), 0, np.cos(yaw)]
#     ])
#
#     # 绕Z轴旋转（roll）
#     Rz = np.array([
#         [np.cos(roll), -np.sin(roll), 0],
#         [np.sin(roll), np.cos(roll), 0],
#         [0, 0, 1]
#     ])
#
#     return Rz @ Ry @ Rx
#
# def align_face(image, pitch, yaw, roll, f=1000):
#     """
#     使用欧拉角进行人脸对齐
#     :param image: 输入图像
#     :param pitch: 俯仰角（绕X轴）
#     :param yaw: 偏航角（绕Y轴）
#     :param roll: 翻滚角（绕Z轴）(hopenet传入的roll角为负值，需要取反)
#     :param f: 虚拟焦距
#     """
#     h, w = image.shape[:2]
#
#     # 定义三维坐标点（以图像中心为原点）
#     src_3d = np.array([
#         [-w/2, -h/2, 0],
#         [w/2, -h/2, 0],
#         [w/2, h/2, 0],
#         [-w/2, h/2, 0]
#     ], dtype=np.float32)
#
#     # 获取旋转矩阵
#     R = _euler_to_rotation_matrix(pitch, yaw, -roll) # roll角取反
#
#     # 应用旋转
#     rotated_3d = src_3d @ R.T  # 矩阵乘法
#
#     # 投影到2D平面（透视投影）
#     dst_2d = []
#     for (X, Y, Z) in rotated_3d:
#         scale = f / (Z + f + 1e-6)  # 防止除以零
#         x = X * scale + w/2
#         y = Y * scale + h/2
#         dst_2d.append([x, y])
#
#     # 原始图像四点坐标
#     src_points = np.array([
#         [0, 0],
#         [w, 0],
#         [w, h],
#         [0, h]
#     ], dtype=np.float32)
#
#     # 计算透视变换矩阵
#     M, _ = cv2.findHomography(src_points, np.array(dst_2d, dtype=np.float32))
#
#     # 应用透视变换
#     aligned = cv2.warpPerspective(image, M, (w, h))
#     return aligned
#
# # 写一个批量对齐人脸的函数
# def align_faces_batch(images, poses):
#     """
#     批量对齐人脸
#     :param images: 输入图像列表
#     :param poses: 姿态参数列表，包含 (pitch, yaw, roll)
#     """
#     aligned_faces = []
#     for img, (pitch, yaw, roll) in zip(images, poses):
#         aligned_face = align_face(img, pitch, yaw, roll)
#         aligned_faces.append(aligned_face)
#     return aligned_faces
#
# # 使用示例
# if __name__ == "__main__":
#     # 加载图像
#     img = cv2.imread("input.jpg")
#
#     # 假设检测到的欧拉角（单位：度）
#     pitch = -10  # 俯仰角（低头）
#     yaw = 15     # 偏航角（向右转头）
#     roll = 5     # 翻滚角（顺时针倾斜）
#
#     # 进行对齐
#     aligned_img = align_face(img, pitch, yaw, roll)
#
#     # 显示结果
#     cv2.imshow("Original", img)
#     cv2.imshow("Aligned", aligned_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()