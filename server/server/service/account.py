from pathlib import Path
from algorithm.base import *
from algorithm.face_pose_estimation import face_pose_estimate_single

def add_account(account) -> dict:
    # 创建一个新的账户文件夹
    account_path = Path(facebank_path) / account
    if not account_path.exists():
        account_path.mkdir(parents=True, exist_ok=True)
        print(f'Account {account} created.')
        return str(account_path)
    else:
        print(f'Account {account} already exists.')
        raise ValueError("Account already exists.")

def add_account_facebank(account, file_name, face, model=hopenetlite) -> str:
    yaw, pitch, row =face_pose_estimate_single(model, face)
    # 三者的绝对值都小于10度
    if not (abs(yaw) < 10 and abs(pitch) < 10 and abs(row) < 10):
        raise ValueError(f"Face pose is not frontal enough. yaw={yaw}, pitch={pitch}, roll={row}")
    facebank_dir = os.path.join(facebank_path, account, file_name)
    cv2.imwrite(os.path.join(facebank_dir, file_name), face)

    facebank_map[account] = prepare_facebank(facebank_dir, model, force_rebuild=True)
    return facebank_dir