import cv2

# 编写一个从多个人脸区域boxes中截取人脸的函数
def crop_faces(image, boxes) -> list:
    faces = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        faces.append(image[y1:y2, x1:x2])
    return faces

def read_image_rgb(path):
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def singleton(cls):
    """单例装饰器"""
    _instances = {}

    def wrapper(*args, **kwargs):
        if cls not in _instances:
            _instances[cls] = cls(*args, **kwargs)
        return _instances[cls]

    return wrapper


@singleton
class VideoSource:

    def __init__(self):
        # 初始化时没有视频源
        self._current_source = None
        self._source_map = {}
        self._cap = None

    def configure(self, source):
        """配置视频源（支持摄像头索引或文件路径）"""
        if source in self._source_map:
            self._current_source = source
            self._cap = self._source_map[source]
            return  # 源未改变无需重新初始化

        # # 释放现有资源
        # self.release()

        # 创建新的视频捕获对象
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise ValueError(f"无法打开视频源: {source}")
        self._current_source = source
        self._source_map[source] = self._cap

    def read_frame(self):
        """读取当前帧"""
        if self._cap is None:
            raise RuntimeError("视频源未初始化")
        return self._cap.read()

    def is_active(self):
        """检查视频源状态"""
        return self._cap is not None and self._cap.isOpened()

    def release(self, source=None):
        if source is None:
            for cap in self._source_map.values():
                cap.release()
        else:
            if source in self._source_map:
                self._source_map[source].release()
                del self._source_map[source]



def get_video_handler(source):
    """获取配置好的视频处理器单例"""
    handler = VideoSource()
    handler.configure(source)
    return handler


# 使用示例
if __name__ == "__main__":
    # 测试摄像头（索引0）
    cam = get_video_handler(0)
    try:
        while cam.is_active():
            ret, frame = cam.read_frame()
            if not ret: break
            cv2.imshow('Live', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    # 测试视频文件
    video = get_video_handler("demo.mp4")
    try:
        while video.is_active():
            ret, frame = video.read_frame()
            if not ret: break
            cv2.imshow('Recording', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    finally:
        video.release()
        cv2.destroyAllWindows()