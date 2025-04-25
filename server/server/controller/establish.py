import asyncio

from ninja import UploadedFile, File
from ninja_extra import NinjaExtraAPI, api_controller, http_get, ControllerBase
from django.http import StreamingHttpResponse

establish = NinjaExtraAPI()

@establish.post('')
async def upload_images(request, files: list[UploadedFile] = File(...)):
    async def sse_event_generator():
        print('开始处理文件')
        # 初始化进度
        total_files = len(files)

        # 处理每个文件并发送事件
        for index, file in enumerate(files, 1):
            print(f'正在处理第 {index} 个文件：{file.name}')
            # 模拟处理时间（替换为实际处理逻辑）
            await asyncio.sleep(2)

            # 构造 SSE 格式数据
            progress = f"{index}/{total_files}"
            yield f"data: 正在处理第 {progress} 个文件：{file.name}\n\n"

            # 这里添加实际图片处理逻辑
            # processed_data = process_image(file)

            # 发送处理结果
            yield f"data: 文件 {file.name} 处理完成\n\n"

        # 结束信号
        yield "data: [END]\n\n"


    return StreamingHttpResponse(
        sse_event_generator(),
        content_type="text/event-stream",
        headers={'Cache-Control': 'no-cache'}
    )





