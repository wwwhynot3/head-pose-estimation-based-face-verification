# import cv2
# cap = cv2.VideoCapture("rtsp://127.0.0.1")
# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret:
#         cv2.imshow("frame", frame)
from server.service.processor import process_frame
import cv2
if __name__ == "__main__":
    pic = cv2.imread('resources/pictures/input/1.jpeg')
    frame, result, score = process_frame(pic)
    for i, (res, sc) in enumerate(zip(result, score)):
        print(f"Image {i}: {res} ({sc:.2f})")
    cv2.imshow("frame", frame)
    