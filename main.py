import time
from typing import Union

import numpy as np

from algorithm.face_alignment.onet import LiteONet
from algorithm.face_detection.prcnn import PRCNN

from fastapi import FastAPI
from algorithm.face_detection import *
from algorithm.hopenet import *
from algorithm.utils import *
import cv2
import torchvision
import torch
from PIL import Image
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor
from algorithm.hopenet.hopenet import HopeNet
from algorithm.face_detection.prcnn import PRCNN
from algorithm.face_alignment import *

device = 'cpu'
hopenet_model_path = 'resources/model/hopenet.pkl'
# -------------------- 初始化全局模型 --------------------
def init_models():
    # 实际项目中需加载预训练权重
    detector = PRCNN(image_size=160, thresholds=[0.8, 0.9], device=device, min_face_size=40)
    pose_net = HopeNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    pose_net.load_state_dict(torch.load(hopenet_model_path, map_location=device))
    pose_net = pose_net.to(device)
    pose_net.eval()
    return detector, pose_net


# -------------------- 图像预处理管道 --------------------
def create_pipelines():
    # 人脸检测预处理
    rgb_transform = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    # 姿态估计预处理
    pose_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return rgb_transform, pose_transform


# -------------------- 并行处理单元 --------------------
def process_face(face_region, pose_net, pose_transform):
    """ 并行处理单个人脸姿态估计 """
    with torch.no_grad():
        # 转换颜色空间并应用预处理
        # img_pil = Image.fromarray(face_roi)
        input_tensor = pose_transform(face_region).unsqueeze(0)
        # 执行推理
        yaw, pitch, roll = pose_net(input_tensor)
        # 转换欧拉角
    # idx_tensor = torch.FloatTensor([i for i in range(66)]).to(device)
    yaw_pred = utils.softmax_temperature(yaw.data, 1)
    pitch_pred = utils.softmax_temperature(pitch.data, 1)
    roll_pred = utils.softmax_temperature(roll.data, 1)
    idx_tensor = torch.FloatTensor(range(66), device=device)
    yaw_deg = (torch.sum(yaw_pred * idx_tensor) * 3 - 99).item()
    pitch_deg = (torch.sum(pitch_pred * idx_tensor) * 3 - 99).item()
    roll_deg = (torch.sum(roll_pred * idx_tensor) * 3 - 99).item()
    return yaw_deg, pitch_deg, roll_deg


# -------------------- 主处理函数 --------------------
def face_pose_pipeline(img_path, output_path):
    # 初始化模型和预处理
    detector, pose_net = init_models()
    rgb_transform, pose_transform = create_pipelines()

    # 读取并预处理图像
    frame = cv2.imread(img_path)
    # rgb_frame = rgb_transform(frame)

    # 执行人脸检测
    boxes, probs = detector.detect(frame)
    pic_index = 0
    # 并行处理所有人脸
    with ThreadPoolExecutor() as executor:
        futures = []
        face_coords = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face_roi = frame[y1:y2, x1:x2]
            # cv2.imwrite(f'resources/pictures/output/{pic_index}.jpeg', face_roi)
            pic_index += 1
            face_roi = rgb_transform(face_roi)
            # 提交并行任务
            future = executor.submit(
                process_face,
                face_roi,
                pose_net,
                pose_transform
            )
            futures.append(future)
            face_coords.append((x1, y1, x2, y2))
    for (box, future) in zip(face_coords, futures):
        yaw, pitch, roll = future.result()
        print('yaw:', yaw, 'pitch:', pitch, 'roll:', roll)
        tdx, tdy = calculate_td(box)
        draw_axis(frame, yaw, pitch, roll, tdx, tdy, (box[2] - box[0])/2)
    # 收集结果并绘制
    # for (x1, y1, x2, y2), future in zip(face_coords, futures):
    #     yaw, pitch, roll = future.result()
    #
    #     # 绘制边界框
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #
    #     # 绘制姿态文本
    #     text = f'Y:{yaw:.1f}, P:{pitch:.1f}, R:{roll:.1f}'
    #     cv2.putText(frame, text, (x1, y1 - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 保存结果
    cv2.imwrite(output_path, frame)
    print(f'Result saved to {output_path}')

def face_align():
    img = cv2.imread('resources/pictures/input/1-2.jpeg')
    yaw = 25.51507568359375
    pitch = 19.894866943359375
    roll = -15.88824462890625
    # yaw = 0
    # pitch = 0
    # roll = 30
    result = align_face(img, -pitch, yaw, -roll)

    # 显示结果
    # cv2.imshow("Original", img)
    # cv2.imshow("Euler Corrected", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite('resources/pictures/output/1-2-out.jpeg', result)
def face_point():
    # 读取并处理图像
    onet = LiteONet()
    img = cv2.imread("resources/pictures/input/1-2.jpeg")
    # face = cv2.resize(img, (48, 48))
    # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    # h, w = face.shape[:2]
    # print(face.shape)
    # landmarks , _ = detect_landmarks([face],onet, return_relative=True, normalize=True)
    # print(landmarks)
    # landmarks = landmarks[0] * np.array([w, h])
    # print(landmarks)
    # # 绘制关键点
    # for x, y in landmarks:
    #     cv2.circle(face, (x, y), 1, (0, 255, 0), -1)
    img = get_landmarks(img, onet)
    cv2.imwrite("resources/pictures/output/1-2-landmarks.jpeg", img[0])
# -------------------- 执行示例 --------------------
if __name__ == "__main__":
    # face_pose_pipeline('resources/pictures/input/1-1.jpeg', 'resources/pictures/output/test_pr_hopenet_1-1.jpeg')
    # face_align()
    face_point()





# app = FastAPI()
#
#
# @app.get("/")
# def read_root():
#     return {"Hello": "World"}
#
#
# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}
# def test_pr_hopenet():
#     device = 'cpu'
#     frame = cv2.imread('resources/pictures/input/1.jpeg')
#     cnn = PRCNN()
#     boxes, probs = cnn.detect(frame)
#     faces = crop_faces(frame, boxes)
#     model =  HopeNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
#     model.config('resources/model/hopenet.pkl',device)
#     transform = get_hopenet_transformer()
#     if faces is not None:
#         for face in faces:
#             cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#             input_img = transform(face).unsqueeze(0)
#             with torch.no_grad():
#                 images = Variable(input_img).to(device)  # 替换Variable(input_img.cuda())
#                 yaw, pitch, roll = model(images)
#             idx_tensor = torch.FloatTensor([i for i in range(66)]).to(device)
#             yaw_pred = utils.softmax_temperature(yaw.data, 1)
#             pitch_pred = utils.softmax_temperature(pitch.data, 1)
#             roll_pred = utils.softmax_temperature(roll.data, 1)
#
#             yaw_deg = (torch.sum(yaw_pred * idx_tensor, 1).cpu()[0] * 3 - 99).item()
#             pitch_deg = (torch.sum(pitch_pred * idx_tensor, 1).cpu()[0] * 3 - 99).item()
#             roll_deg = (torch.sum(roll_pred * idx_tensor, 1).cpu()[0] * 3 - 99).item()
#             print(yaw_deg, pitch_deg, roll_deg)
#             # 6. 绘制姿态轴并保存
#             img = utils.draw_axis(img, yaw_deg, pitch_deg, roll_deg)
#             res = cv2.imwrite("resources/pictures/output/1_5_out.jpeg", img)
#
#
# if __name__ == '__main__':
#     from algorithm.face_detection import example
    from algorithm.hopenet import test_hopenet
#     # example.test_pr()
#     # time.sleep(10)
#     test_hopenet()