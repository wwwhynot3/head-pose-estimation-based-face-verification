import cv2

from algorithm import hopenetlite, mobilefacenet
from algorithm.face_detection import detect_face
from algorithm.face_pose_estimation import face_pose_estimate_batch
from algorithm.face_alignment import align_faces_batch
from algorithm import face_recognition_batch
from algorithm.base import facebank_default_account




def process_frame(frame, account = facebank_default_account ,min_probs=0.7, face_threshold=0.4):
    """
    输入的frame请为RGB格式,
    输出的frame也为RGB格式
    """
    # print(f"Processing frame for account {account}...")
    boxes, faces, probs = detect_face(frame, min_prob=min_probs)
    if len(boxes) == 0:
        return frame, [], []
    poses = face_pose_estimate_batch(hopenetlite, faces)
    aligned_faces = align_faces_batch(faces, poses)
    results, scores = face_recognition_batch(image_batch=aligned_faces, threshold=face_threshold, model=mobilefacenet, account=account)
    for (face, result) in zip(boxes, results):
        x1, y1, x2, y2 = map(int, face)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{result}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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

