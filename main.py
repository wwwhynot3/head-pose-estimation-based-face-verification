import time

import cv2

from algorithm import face_pose_estimate_single
from algorithm.face_pose_estimation_cp1 import *
import torchvision
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor
from algorithm.model.hopenet import HopeNet
from algorithm.model.prcnn import PRCNN
from algorithm.face_alignment import *
from algorithm.base import *

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
    onet = KeypointNet()
    onet.eval()
    imgs = [cv2.imread('resources/pictures/input/1_0.jpeg')]
    land = detect_face(imgs.copy(), onet, device)
    # 打印结果
    for i, kpts in enumerate(land):
        print(f"人脸{i + 1}关键点：")
        print(f"左眼: ({kpts[0][0]:.10f}, {kpts[0][1]:.10f})")
        print(f"右眼: ({kpts[1][0]:.10f}, {kpts[1][1]:.10f})")
        print(f"鼻子: ({kpts[2][0]:.10f}, {kpts[2][1]:.10f})")
        print(f"左嘴角: ({kpts[3][0]:.10f}, {kpts[3][1]:.10f})")
        print(f"右嘴角: ({kpts[4][0]:.10f}, {kpts[4][1]:.10f})")
        print()
    # # print(land)
    for img, points in zip(imgs, land):
        for point in points:
            x = int(point[0])
            y = int(point[1])
            print(point)
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    cv2.imwrite("resources/pictures/output/1_0_landmarks.jpeg", imgs[0])

    # 返回形状 (5,2) 的tensor，对应5个关键点坐标

    # 示例：批量检测
    # batch_input = [img1, img2, img3]  # 多个已裁剪的人脸图像
    # batch_points = detector.detect(batch_input)  # 形状 (3,5,2)
    # img = cv2.imread("resources/pictures/input/1-2.jpeg")
    #
    # cv2.imwrite("resources/pictures/output/1-2-landmarks.jpeg", img[0])
# -------------------- 执行示例 --------------------

# def recognition():
#     from algorithm.face_recognition.example import face_recognition_pipeline
#     face_recognition_pipeline(threshold=1)
def testt_hopenet(model, pic):
    # 4. 推理预测
    with torch.no_grad():
        images = pic.to(device)  # 替换Variable(input_img.cuda())
        yaw, pitch, roll = model(images)

    # 5. 转换欧拉角[1](@ref)
    # idx_tensor = torch.FloatTensor([i for i in range(66)]).cuda(args.gpu_id)
    idx_tensor = torch.FloatTensor([i for i in range(66)]).to(device)
    yaw_pred = utils.softmax_temperature(yaw.data, 1)
    pitch_pred = utils.softmax_temperature(pitch.data, 1)
    roll_pred = utils.softmax_temperature(roll.data, 1)

    yaw_deg = (torch.sum(yaw_pred * idx_tensor, 1).cpu()[0] * 3 - 99).item()
    pitch_deg = (torch.sum(pitch_pred * idx_tensor, 1).cpu()[0] * 3 - 99).item()
    roll_deg = (torch.sum(roll_pred * idx_tensor, 1).cpu()[0] * 3 - 99).item()
    return yaw_deg, pitch_deg, roll_deg
    
def compare_hopenet():
    img = cv2.imread("resources/pictures/input/1_0.jpeg")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
    input_img = hopenet_transform(img_rgb).unsqueeze(0)  # 增加batch维度

    yaw_deg, pitch_deg, roll_deg = testt_hopenet(hopenet, input_img)
    # 6. 绘制姿态轴并保存
    img1 = utils.draw_axis(img.copy(), yaw_deg, pitch_deg, roll_deg)
    res = cv2.imwrite("resources/pictures/output/1_0_out1.jpeg", img1)
    
    yaw_deg, pitch_deg, roll_deg = testt_hopenet(shuffledhopenet, input_img)
    # 6. 绘制姿态轴并保存
    img2 = utils.draw_axis(img.copy(), yaw_deg, pitch_deg, roll_deg)
    res = cv2.imwrite("resources/pictures/output/1_0_out2.jpeg", img2)
    print(f'write success: {res}')
    print('done')
    
def align():
    img = cv2.imread("resources/pictures/input/1_0.jpeg")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
    input_img = hopenet_transform(img_rgb).unsqueeze(0)  # 增加batch维度

    yaw_deg, pitch_deg, roll_deg = testt_hopenet(hopenet, input_img)
    mm = align_face(image=img, pitch=pitch_deg, yaw=yaw_deg, roll=roll_deg)
    cv2.imwrite("resources/pictures/output/1_0_out2_align.jpeg", mm)

def batch_hopenet1():
    from algorithm.face_pose_estimation_cp1 import batch_pose_estimate, draw_axis
    from algorithm.face_pose_estimation_cp2 import face_pose_estimate_batch
    from algorithm.face_pose_estimation_og import face_pose_estimate_single
    from algorithm.face_detection import detect_face
    from algorithm.base import prcnn, shuffledhopenet, hopenet
    img = cv2.imread("resources/pictures/input/1.jpeg")
    faces, probs = detect_face(img.copy(), min_prob=0.8)
    count = 0
    for face in faces:
        yaw, pitch, roll = face_pose_estimate_single(hopenet, face)
        print(yaw, pitch, roll)
        face = face.copy()
        face1 = face.copy()
        draw_axis(face, yaw, pitch, roll)
        cv2.imwrite(f"resources/pictures/output/1_{count}_out_og_hopenet.jpeg", face)
        yaw, pitch, roll = face_pose_estimate_single(shuffledhopenet, face)
        print(yaw, pitch, roll)
        draw_axis(face1, yaw, pitch, roll)
        cv2.imwrite(f"resources/pictures/output/1_{count}_out_og_shuffledhopenet.jpeg", face1)
        count += 1

    print("----------------")
    # 截取时间
    start = time.time()
    res1 = batch_pose_estimate(hopenet, faces)
    # 计算耗时
    end = time.time()
    print(f"batch_pose_estimate_hopenet time: {end - start:.4f} seconds")
    start = time.time()
    res3 = batch_pose_estimate(shuffledhopenet, faces)
    end = time.time()
    print(f"batch_pose_estimate_shuffledhopenet time: {end - start:.4f} seconds")
    count = 0
    for face, (yaw, pitch, roll), (yaw2, pitch2, roll2) in zip(faces, res1, res3):
        # yaw_deg = (torch.sum(yaw_batch * idx_tensor) * 3 - 99).item()
        # pitch_deg = (torch.sum(pitch_batch * idx_tensor) * 3 - 99).item()
        # roll_deg = (torch.sum(roll_batch * idx_tensor) * 3 - 99).item()
        print(yaw, pitch, roll)
        face = face.copy()
        face1 = face.copy()
        draw_axis(face, yaw, pitch, roll)
        cv2.imwrite(f"resources/pictures/output/1_{count}_out_cp1_hopenet.jpeg", face)
        draw_axis(face1, yaw2, pitch2, roll2)
        cv2.imwrite(f"resources/pictures/output/1_{count}_out_cp1_shuffledhopenet.jpeg", face1)
        count += 1

    print("---------------")

    count = 0
    start = time.time()
    res2 = face_pose_estimate_batch(hopenet, faces)  #看起来比1对
    end = time.time()
    print(f"face_pose_estimate_batch_hopenet time: {end - start:.4f} seconds")
    start = time.time()
    res4 = face_pose_estimate_batch(shuffledhopenet, faces)
    end = time.time()
    print(f"face_pose_estimate_batch_shuffledhopenet time: {end - start:.4f} seconds")
    for face, (yaw, pitch, roll), (yaw2, pitch2, roll2) in zip(faces, res2, res4):
        # yaw_deg = (torch.sum(yaw_batch * idx_tensor) * 3 - 99).item()
        # pitch_deg = (torch.sum(pitch_batch * idx_tensor) * 3 - 99).item()
        # roll_deg = (torch.sum(roll_batch * idx_tensor) * 3 - 99).item()
        print(yaw, pitch, roll)
        face = face.copy()
        face1 = face.copy()
        draw_axis(face, yaw, pitch, roll)
        cv2.imwrite(f"resources/pictures/output/1_{count}_out_cp2_hopenet.jpeg", face)
        draw_axis(face1, yaw2, pitch2, roll2)
        cv2.imwrite(f"resources/pictures/output/1_{count}_out_cp2_shuffledhopenet.jpeg", face1)
        count += 1

    
if __name__ == "__main__":
    # face_pose_pipeline('resources/pictures/input/1-1.jpeg', 'resources/pictures/output/test_pr_hopenet_1-1.jpeg')
    # face_align()
    # face_point()
    # recognition()
    # compare_hopenet()
    # align()
    batch_hopenet1()





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
#     model.config('resources/model/face_pose_estimation.bak.pkl',device)
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
#     from algorithm.face_pose_estimation.bak import test_hopenet
#     example.test_pr()
#     # time.sleep(10)
#     test_hopenet()