import time

import cv2

import algorithm
from algorithm import hopenetlite, mobilefacenet, font_size
from algorithm.face_detection import detect_face
from algorithm.face_pose_estimation import face_pose_estimate_batch
from algorithm.face_alignment_euler import align_faces_batch
from algorithm.face_alignment_landmark import align_faces_batch
from algorithm import face_recognition_batch
from algorithm.base import facebank_default_account, cv2PutChineseText


def get_face(frame):
    """
    输入的frame请为RGB格式,
    输出的frame也为RGB格式
    """
    # print(f"Processing frame for account {account}...")
    boxes, faces, probs = detect_face(frame, min_prob=0.9)
    if len(boxes) != 1:
        raise ValueError(f"检测到{len(boxes)}张人脸，请确保只输入一张人脸")
    return faces[0]

def process_frame(frame, account = facebank_default_account):
    """
    输入的frame请为RGB格式,
    输出的frame也为RGB格式
    """
    # print(f"Processing frame for account {account}...")
    boxes, faces, probs, landmarks = detect_face(frame, min_prob=0.9, landmark=True)
    if len(boxes) == 0:
        return frame, [], []
    poses = face_pose_estimate_batch(hopenetlite, faces)
    aligned_faces = algorithm.face_alignment_euler.align_faces_batch(faces, poses)
    timestamp = time.time()
    cv2.imwrite(f'resources/upload/euler_{timestamp}.jpg', aligned_faces[0])
    aligned_faces =  algorithm.face_alignment_landmark.align_faces_batch(aligned_faces, boxes, landmarks)
    cv2.imwrite(f'resources/upload/landmark_{timestamp}.jpg', aligned_faces[0])
    results, scores = face_recognition_batch(image_batch=aligned_faces, threshold=0.4, model=mobilefacenet, account=account)
    # frame = frame.copy()
    # 防止内存可读性导致的错误
    if not frame.flags.writeable:
        frame = frame.copy()
    for (face, result, (yaw, pitch, row)) in zip(boxes, results, poses):
        x1, y1, x2, y2 = map(int, face)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 在人脸框右方分三行显示人脸角度
        text = f"Yaw: {yaw:.2f}°\nPitch: {pitch:.2f}°\nRoll: {row:.2f}°"
        cv2.putText(frame, f"Yaw: {yaw:.2f}", (x2, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, f"Pitch: {pitch:.2f}", (x2, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, f"Roll: {row:.2f}", (x2, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        # cv2.putText(frame, text, (x2 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        frame = cv2PutChineseText(frame, f"{result}", (x1, y1 - font_size))
    return frame, results, scores

# def test():
#     # 测试代码
#     cap = cv2.VideoCapture(0)  # 使用摄像头
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # 处理每一帧
#         process_frame(frame)
#
#         # 显示原始帧
#         cv2.imshow("Frame", frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()

def test():
    # cv2.imread('resources/pictures/input/1.jpeg')
    # 测试代码
    process_frame(cv2.imread('resources/pictures/input/1.jpeg'))

