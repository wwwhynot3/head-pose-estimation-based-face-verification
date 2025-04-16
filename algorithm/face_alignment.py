import cv2
import numpy as np

def euler_to_rotation_matrix(yaw, pitch, roll):
    # Convert degrees to radians
    yaw = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)
    roll = np.deg2rad(roll)

    # Calculate rotation matrix
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch)],
                    [0, np.sin(pitch), np.cos(pitch)]])

    R_y = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                    [0, 1, 0],
                    [-np.sin(yaw), 0, np.cos(yaw)]])

    R_z = np.array([[np.cos(roll), -np.sin(roll), 0],
                    [np.sin(roll), np.cos(roll), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def align_face(image, yaw, pitch, roll):
    h, w = image.shape[:2]
    R = euler_to_rotation_matrix(yaw, pitch, roll)

    # Define the center of the image
    center = (w // 2, h // 2)

    # Get the affine transformation matrix
    affine_matrix = R[:2, :2]
    translation_vector = R[:2, 2]

    # Apply the affine transformation
    aligned_image = cv2.warpAffine(image, affine_matrix, (w, h), flags=cv2.INTER_LINEAR)

    return aligned_image


# Example usage
# image = cv2.imread('path_to_image.jpg')
# yaw, pitch, roll = 10, 20, 30  # Example Euler angles
# aligned_image = align_face(image, yaw, pitch, roll)
#
# cv2.imshow('Aligned Face', aligned_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# import cv2
# import numpy as np
#
#
# def euler_to_rotation_matrix(pitch, yaw, roll):
#     """将欧拉角转换为旋转矩阵[8](@ref)"""
#     # 转换为弧度
#     pitch = np.radians(pitch)
#     yaw = np.radians(yaw)
#     roll = np.radians(roll)
#
#     # 各轴旋转矩阵
#     Rx = np.array([[1, 0, 0],
#                    [0, np.cos(pitch), -np.sin(pitch)],
#                    [0, np.sin(pitch), np.cos(pitch)]])
#
#     Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
#                    [0, 1, 0],
#                    [-np.sin(yaw), 0, np.cos(yaw)]])
#
#     Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
#                    [np.sin(roll), np.cos(roll), 0],
#                    [0, 0, 1]])
#
#     # 组合旋转矩阵（顺序：Yaw->Pitch->Roll）
#     R = Rz @ Ry @ Rx
#     return R[:2, :2]  # 取前两行用于仿射变换
#
#
# def composite_transform(image, pitch, yaw, roll, face_points):
#     """
#     复合变换流程：
#     1. 仿射变换矫正平面旋转
#     2. 透视变换校正三维姿态
#     """
#     # 步骤1：仿射变换（矫正平面旋转）[1,4](@ref)
#     R_inv = euler_to_rotation_matrix(-roll, -yaw, -pitch)  # 反向旋转
#     M_affine = np.hstack([R_inv, [[0], [0]]])  # 构建仿射矩阵
#
#     # 应用仿射变换
#     rows, cols = image.shape[:2]
#     affine_img = cv2.warpAffine(image, M_affine, (cols, rows),
#                                 flags=cv2.INTER_LINEAR,
#                                 borderMode=cv2.BORDER_REPLICATE)
#
#     # 步骤2：透视变换（矫正三维姿态）[7,8](@ref)
#     # 定义目标标准点（假设标准正面人脸坐标）
#     dst_points = np.float32([[cols * 0.3, rows * 0.3],  # 左眼
#                              [cols * 0.7, rows * 0.3],  # 右眼
#                              [cols * 0.3, rows * 0.7],  # 左嘴角
#                              [cols * 0.7, rows * 0.7]])  # 右嘴角
#
#     # 转换已矫正后的关键点坐标
#     adjusted_points = cv2.transform(np.array([face_points]), M_affine)[0]
#
#     # 计算透视变换矩阵
#     M_perspective, _ = cv2.findHomography(adjusted_points, dst_points,
#                                           cv2.RANSAC, 5.0)
#
#     # 应用透视变换
#     result = cv2.warpPerspective(affine_img, M_perspective, (cols, rows),
#                                  flags=cv2.INTER_CUBIC)
#
#     return result
#
#
# # 使用示例
# if __name__ == "__main__":
#     # 输入参数
#     image = cv2.imread("face.jpg")
#     pitch = 15  # 俯仰角（度）
#     yaw = -10  # 偏航角（度）
#     roll = 5  # 翻滚角（度）
#
#     # 假设通过人脸检测获取的四个关键点坐标（示例值）
#     face_points = np.float32([[120, 80],  # 左眼
#                               [280, 90],  # 右眼
#                               [130, 300],  # 左嘴角
#                               [270, 310]])  # 右嘴角
#
#     # 执行复合变换
#     output = composite_transform(image, pitch, yaw, roll, face_points)
#
#     # 显示结果
#     cv2.imshow("Original", image)
#     cv2.imshow("Corrected", output)
#     cv2.waitKey(0)