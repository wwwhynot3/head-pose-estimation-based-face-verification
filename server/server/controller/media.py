from io import BytesIO
import cv2
from django.http import FileResponse, HttpResponse
from ninja import Form, NinjaAPI, File
import numpy as np
from ninja.files import UploadedFile
from server.service.processor import process_frame

media_api = NinjaAPI(urls_namespace="media")


@media_api.post('picture')
def rec_face(request, account: str = Form(...), file:UploadedFile = File(...)):
    """
    Recognize a Face Picture
    """
    try:
        # 读取图片
        img = file.read()
        file_name = file.name;
        # 转换为numpy数组
        nparr = np.frombuffer(img, np.uint8)
        # 解码为图像
        face = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(f'resources/upload/{file.name}', face)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        # 添加到facebank
        frame, results, scores = process_frame(face)
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR, frame)
        cv2.imwrite(f'resources/upload/after_{file.name}', frame)
        success, frame_res = cv2.imencode('.jpg', frame)
        frame_bytes = frame_res.tobytes()
        # buffer = BytesIO(frame_bytes)
        # buffer.seek(0)
        # return FileResponse(buffer, content_type="image/jpeg", filename=f"after_{file.name}",  as_attachment=True)
        return HttpResponse(
            frame_bytes,
            content_type="image/jpeg",
            headers={"Content-Disposition": 'inline; filename="processed.jpg"'}
        )
    except Exception as e:
        # return Response.error(f"Error: {str(e)}")
        # 删除注册失败的人脸
        print(f"Error: {str(e)}")
        # raise e
        return {"code": 400, "data": str(e)}
    return {"code":200, "data": account_path}