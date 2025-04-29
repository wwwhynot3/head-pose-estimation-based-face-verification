import numpy as np

from algorithm.model.prcnn import PRCNN
from algorithm.base import prcnn
import cv2

def _crop_faces(image, boxes, probs, min_prob) -> tuple[list, list, list]:
    if boxes.size == 0 or probs.size == 0:
        return [], [], []
    faces = []
    probss = []
    valid_boxes = []
    for box, prob in zip(boxes, probs):
        if prob < min_prob:
            continue
        x1, y1, x2, y2 = map(int, box)
        valid_boxes.append(box)
        faces.append(image[y1:y2, x1:x2])
        probss.append(prob)
    return valid_boxes, faces, probss

def detect_face(frame, min_prob=0) -> tuple[list, list, list]:
    boxes, probs = prcnn.detect(frame)
    boxes = np.array(boxes) if boxes is not None else np.array([])
    probs = np.array(probs) if probs is not None else np.array([])
    return _crop_faces(frame, boxes, probs, min_prob)

def test_pr():
    print('start')
    cnn = PRCNN(image_size=160, thresholds=[0.8, 0.9], device='cuda',min_face_size=40)
    print(f'model loaded success {cnn is not None}')
    frame = cv2.imread('resources/pictures/input/1.jpeg')
    print(f'frame read success: {frame is not None}')
    boxes, probs = cnn.detect(frame)
    i = 0
    for box, prob in zip(boxes, probs):
        x1, y1, x2, y2 = map(int, box)
        res = cv2.imwrite(f'resources/pictures/output/1_{i}out.jpeg', frame[y1:y2, x1:x2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{prob:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        i += 1
        # print(f'write success: {res}')
    # cv2.imwritemulti('resources/pictures/output/1_out.tiff', crop_faces(frame, boxes))
    res = cv2.imwrite('resources/pictures/output/1_out.jpeg', frame)
    print(f'write success: {res}')
    print('done')

# def test_mtcnn():
#     from facenet_pytorch import MTCNN
#     cnn = MTCNN(image_size=160, thresholds=[0.6, 0.7, 0.7], device='cuda',min_face_size=40)
#     frame = cv2.imread('resources/pictures/1.jpeg')
#     boxes, probs = cnn.detect(frame)
#     for box, prob in zip(boxes, probs):
#         x1, y1, x2, y2 = map(int, box)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f'{prob:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.imwrite('resources/pictures/1_1_out.jpeg', frame)
if __name__ == '__main__':
    test_pr()
