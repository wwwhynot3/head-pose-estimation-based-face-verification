from algorithm.face_detection import PRCNN
import  cv2
def test_pr():
    cnn = PRCNN(image_size=160, thresholds=[0.8, 0.9], device='cuda',min_face_size=40)
    frame = cv2.imread('resources/pictures/1.jpeg')
    boxes, probs = cnn.detect(frame)
    for box, prob in zip(boxes, probs):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{prob:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite('resources/pictures/1_out.jpeg', frame)

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