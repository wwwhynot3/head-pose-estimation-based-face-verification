import cv2

from algorithm import hopenetlite, mobilefacenet
# from algorithm.base import prcnn
from algorithm.face_detection import detect_face
from algorithm.face_pose_estimation import face_pose_estimate_batch
from algorithm.face_alignment import align_faces_batch
from algorithm.face_recognition import face_recognition_batch




def process_frame(frame, min_probs=0.7, face_threshold=0.4):
    # 人脸检测
    faces, probs = detect_face(frame)


    faces, probs = detect_face(frame, min_prob=min_probs)

    print(f"Detected {len(faces)} faces")
    for face, prob in zip(faces, probs):
        print(f"Face probability: {prob:.2f}")

    # 姿态估计
    poses = face_pose_estimate_batch(hopenetlite, faces)

    # 人脸对齐
    aligned_faces = align_faces_batch(faces, poses)

    for i, (face, prob) in enumerate(zip(aligned_faces, probs)):
        cv2.imwrite(f"resources/pictures/output/aligned/AlignedFace{i}.jpg", face)
    # 人脸识别

    results, scores = face_recognition_batch(image_batch=aligned_faces, threshold=face_threshold, model=mobilefacenet)

    for i, (result, score) in enumerate(zip(results, scores)):
        print(f"Image {i}: {result} ({score:.2f})")
    return results, scores

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

