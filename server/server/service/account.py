from pathlib import Path

import cv2

from algorithm.base import *
from algorithm.face_pose_estimation import face_pose_estimate_single, face_pose_estimate_batch
from algorithm.face_detection import detect_face
from algorithm.face_alignment import align_faces_batch
def add_account(account) -> str:
    # 创建一个新的账户文件夹
    account_path = Path(facebank_path) / account
    if not account_path.exists():
        account_path.mkdir(parents=True, exist_ok=True)
        print(f'Account {account} created.')
    else:
        print(f'Account {account} already exists.')
    prepare_facebank(account_path, model=hopenetlite, force_rebuild=True)
    return str(account_path)

def add_account_facebank(account, file_name, face, model=hopenetlite) -> str:
    boxes, faces, probs = detect_face(face, min_prob=0.9)
    cv2.imwrite(f'resources/upload/og_{file_name}', face)
    face = faces[0]
    if len(boxes) != 1:
        raise ValueError(f"检测到{len(boxes)}张人脸，请确保只输入一张人脸")
    euler = face_pose_estimate_batch(model, faces)
    yaw, pitch, row = euler[0]
    # 三者的绝对值都小于10度
    if not (abs(yaw) < 30 and abs(pitch) < 30 and abs(row) < 30):
        raise ValueError(f"人脸角度过大，请上传正脸照片 yaw={yaw}, pitch={pitch}, roll={row}")
    cv2.imwrite(f'resources/upload/detected_{file_name}', face)
    aligned_face = align_faces_batch(faces, euler)[0]
    facebank_dir = os.path.join(facebank_path, account)
    cv2.imwrite(f'resources/upload/aligned_{file_name}', aligned_face)
    cv2.imwrite(str(os.path.join(facebank_dir, file_name)), aligned_face)

    facebank_map[account] = prepare_facebank(facebank_dir, mobilefacenet, force_rebuild=True)
    return facebank_dir