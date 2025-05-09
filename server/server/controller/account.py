import cv2
from ninja import Form, NinjaAPI, File, Query, Schema
import numpy as np
from server.service.account import add_account, add_account_facebank
from typing import List
from ninja.files import UploadedFile
account_api = NinjaAPI(urls_namespace="account")
class RegisterModel(Schema):
    """
    Register model
    """
    account: str
    
@account_api.post('register')
def register_account(request, account: RegisterModel):
    """
    Register a new user.
    """
    # 打印日志
    print(f"Registering account {account.account}")
    try:
        account_path = add_account(account.account)
    except Exception as e:
        return {"code": 400, "data": str(e)}
    return {"code":200, "data": account_path}

@account_api.post('register_face')
def register_face(request, account: str = Form(...), file:UploadedFile = File(...)):
    """
    Register a new face.
    """
    # 打印日志
    print(f"Registering face for account {account}， file name: {file.name}")
    try:
        # 读取图片
        img = file.read()
        # 转换为numpy数组
        nparr = np.frombuffer(img, np.uint8)
        # 解码为图像
        face = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(f'resources/upload/{file.name}', face)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        # 添加到facebank
        file_name = file.name
        account_path = add_account_facebank(account, file_name, face)
    except Exception as e:
        # return Response.error(f"Error: {str(e)}")
        # 删除注册失败的人脸
        print(f"Error: {str(e)}")
        # raise e
        return {"code": 400, "data": str(e)}
    return {"code":200, "data": account_path}
