# import cv2
# cap = cv2.VideoCapture("rtsp://127.0.0.1")
# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret:
#         cv2.imshow("frame", frame)
from server.server.service import test

if __name__ == "__main__":
    test()