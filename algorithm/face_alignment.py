import cv2
import numpy as np

def _euler_to_rotation_matrix(pitch, yaw, roll):
    """将欧拉角转换为3x3旋转矩阵"""
    # 转换为弧度
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    roll = np.radians(roll)

    # 绕X轴旋转（pitch）
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])

    # 绕Y轴旋转（yaw）
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    # 绕Z轴旋转（roll）
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx

def align_face(image, pitch, yaw, roll, f=1000):
    """
    使用欧拉角进行人脸对齐
    :param image: 输入图像
    :param pitch: 俯仰角（绕X轴）
    :param yaw: 偏航角（绕Y轴）
    :param roll: 翻滚角（绕Z轴）(hopenet传入的roll角为负值，需要取反)
    :param f: 虚拟焦距
    """
    h, w = image.shape[:2]
    
    # 定义三维坐标点（以图像中心为原点）
    src_3d = np.array([
        [-w/2, -h/2, 0],
        [w/2, -h/2, 0],
        [w/2, h/2, 0],
        [-w/2, h/2, 0]
    ], dtype=np.float32)

    # 获取旋转矩阵
    R = _euler_to_rotation_matrix(pitch, yaw, -roll) # roll角取反
    
    # 应用旋转
    rotated_3d = src_3d @ R.T  # 矩阵乘法

    # 投影到2D平面（透视投影）
    dst_2d = []
    for (X, Y, Z) in rotated_3d:
        scale = f / (Z + f + 1e-6)  # 防止除以零
        x = X * scale + w/2
        y = Y * scale + h/2
        dst_2d.append([x, y])
    
    # 原始图像四点坐标
    src_points = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)

    # 计算透视变换矩阵
    M, _ = cv2.findHomography(src_points, np.array(dst_2d, dtype=np.float32))
    
    # 应用透视变换
    aligned = cv2.warpPerspective(image, M, (w, h))
    return aligned

# 使用示例
if __name__ == "__main__":
    # 加载图像
    img = cv2.imread("input.jpg")
    
    # 假设检测到的欧拉角（单位：度）
    pitch = -10  # 俯仰角（低头）
    yaw = 15     # 偏航角（向右转头）
    roll = 5     # 翻滚角（顺时针倾斜）
    
    # 进行对齐
    aligned_img = align_face(img, pitch, yaw, roll)
    
    # 显示结果
    cv2.imshow("Original", img)
    cv2.imshow("Aligned", aligned_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()