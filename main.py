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


def batch_hopenet1():
    """
    看起来HopeNetLite的Unofficial实现比ShuffleNetV2的官方实现效果好
    看起来速度 cp2 > cp1 > og
    """
    from algorithm.face_pose_estimation_cp1 import batch_pose_estimate, draw_axis
    from algorithm.face_pose_estimation_cp2 import face_pose_estimate_batch
    from algorithm.face_pose_estimation_og import face_pose_estimate_single
    from algorithm.face_detection import detect_face
    from algorithm.base import prcnn, shuffledhopenet, hopenet
    img = cv2.imread("resources/pictures/input/1.jpeg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces, probs = detect_face(img.copy(), min_prob=0.8)

    count = 0
    time1 = 0
    time2 = 0
    time3 = 0
    for face in faces:
        timee = time.time()
        yaw, pitch, roll = face_pose_estimate_single(hopenet, face)
        time1 += time.time() - timee
        print(yaw, pitch, roll)
        face = face.copy()
        face1 = face.copy()
        face2 = face.copy()
        draw_axis(face, yaw, pitch, roll)
        cv2.imwrite(f"resources/pictures/output/1_{count}_out_og_hopenet.jpeg", face)
        timee = time.time()
        yaw, pitch, roll = face_pose_estimate_single(shuffledhopenet, face)
        time2 += time.time() - timee
        print(yaw, pitch, roll)
        draw_axis(face1, yaw, pitch, roll)
        cv2.imwrite(f"resources/pictures/output/1_{count}_out_og_shuffledhopenet.jpeg", face1)
        timee = time.time()
        yaw, pitch, roll = face_pose_estimate_single(hopenetlite, face2)
        time3 += time.time() - timee
        print(yaw, pitch, roll)
        draw_axis(face2, yaw, pitch, roll)
        cv2.imwrite(f"resources/pictures/output/1_{count}_out_og_hopenetlite.jpeg", face2)
        count += 1
    print(f"face_pose_estimate_single_hopenet time: {time1:.4f} seconds")
    print(f"face_pose_estimate_single_shuffledhopenet time: {time2:.4f} seconds")
    print(f"face_pose_estimate_single_hopenetlite time: {time3:.4f} seconds")
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
    start = time.time()
    res5 = batch_pose_estimate(hopenetlite, faces)
    end = time.time()
    print(f"batch_pose_estimate_hopenetlite time: {end - start:.4f} seconds")
    count = 0
    for face, (yaw, pitch, roll), (yaw2, pitch2, roll2), (yaw3, pitch3, roll3) in zip(faces, res1, res3, res5):
        # yaw_deg = (torch.sum(yaw_batch * idx_tensor) * 3 - 99).item()
        # pitch_deg = (torch.sum(pitch_batch * idx_tensor) * 3 - 99).item()
        # roll_deg = (torch.sum(roll_batch * idx_tensor) * 3 - 99).item()
        print(yaw, pitch, roll)
        print(yaw2, pitch2, roll2)
        print(yaw3, pitch3, roll3)
        face = face.copy()
        face1 = face.copy()
        face2 = face.copy()
        draw_axis(face, yaw, pitch, roll)
        cv2.imwrite(f"resources/pictures/output/1_{count}_out_cp1_hopenet.jpeg", face)
        draw_axis(face1, yaw2, pitch2, roll2)
        cv2.imwrite(f"resources/pictures/output/1_{count}_out_cp1_shuffledhopenet.jpeg", face1)
        draw_axis(face2, yaw3, pitch3, roll3)
        cv2.imwrite(f"resources/pictures/output/1_{count}_out_cp1_hopenetlite.jpeg", face2)
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
    start = time.time()
    res6 = face_pose_estimate_batch(hopenetlite, faces)
    end = time.time()
    print(f"face_pose_estimate_batch_hopenetlite time: {end - start:.4f} seconds")
    for face, (yaw, pitch, roll), (yaw2, pitch2, roll2), (yaw3, pitch3, roll3) in zip(faces, res2, res4, res6):
        # yaw_deg = (torch.sum(yaw_batch * idx_tensor) * 3 - 99).item()
        # pitch_deg = (torch.sum(pitch_batch * idx_tensor) * 3 - 99).item()
        # roll_deg = (torch.sum(roll_batch * idx_tensor) * 3 - 99).item()
        print(yaw, pitch, roll)
        print(yaw2, pitch2, roll2)
        print(yaw3, pitch3, roll3)
        face = face.copy()
        face1 = face.copy()
        face2 = face.copy()
        draw_axis(face, yaw, pitch, roll)
        cv2.imwrite(f"resources/pictures/output/1_{count}_out_cp2_hopenet.jpeg", face)
        draw_axis(face1, yaw2, pitch2, roll2)
        cv2.imwrite(f"resources/pictures/output/1_{count}_out_cp2_shuffledhopenet.jpeg", face1)
        draw_axis(face2, yaw3, pitch3, roll3)
        cv2.imwrite(f"resources/pictures/output/1_{count}_out_cp2_hopenetlite.jpeg", face2)
        count += 1

    """
evaled pnetin.py                                                                                                                                                                                                                                                                ─╯
evaled rnet
17.170589447021484 3.1911277770996094 -14.508636474609375
11.71731948852539 -5.685733795166016 -15.420495986938477
-22.962799072265625 -9.291234970092773 2.519817352294922
1.4910507202148438 -6.070512771606445 1.955657958984375
-6.368053436279297 0.7957649230957031 3.666675567626953
-9.68562126159668 -4.987209320068359 0.010837554931640625
2.028522491455078 5.504722595214844 1.7347869873046875
-26.82344627380371 -12.195236206054688 3.1277389526367188
-5.509008407592773 -5.608829498291016 8.133968353271484
-7.913858413696289 -7.318336486816406 6.507511138916016
25.37651824951172 16.1729736328125 -15.831653594970703
-7.523139953613281 -8.97901725769043 -9.252490997314453
face_pose_estimate_single_hopenet time: 0.0527 seconds
face_pose_estimate_single_shuffledhopenet time: 0.0325 seconds
----------------
batch_pose_estimate_hopenet time: 0.0259 seconds
batch_pose_estimate_shuffledhopenet time: 0.0109 seconds
15.788258 5.066221 -15.488399
-23.66899 -6.414674 3.9046962
-6.9554095 2.7689357 3.0921617
0.81480503 7.1488433 1.1505942
-6.7995067 -3.2303872 7.6436524
24.754414 21.876995 -15.778517
---------------
face_pose_estimate_batch_hopenet time: 0.0125 seconds
face_pose_estimate_batch_shuffledhopenet time: 0.0058 seconds
17.16996 3.1916199 -14.510437
-22.96283 -9.291824 2.5215607
-6.36747 0.79797363 3.66671
2.027916 5.505783 1.7331085
-5.5097504 -5.60672 8.131279
25.376038 16.17508 -15.8333435

    """


    ####
    """
    new
python main.py
evaled pnet
evaled rnet
16.85895538330078 4.386417388916016 -14.910530090332031
12.58871841430664 2.165863037109375 -13.491405487060547
14.657363891601562 3.3149986267089844 -16.276165008544922
7.427284240722656 -5.588533401489258 -20.482698440551758
5.964008331298828 -12.177200317382812 -16.959056854248047
7.537628173828125 -9.090814590454102 -23.098989486694336
-24.91655158996582 -6.491706848144531 3.8903160095214844
-15.410665512084961 -4.623790740966797 -2.0598678588867188
-22.337888717651367 -5.424596786499023 5.247138977050781
-6.466266632080078 0.7829360961914062 3.6133346557617188
-8.240798950195312 -4.324956893920898 -5.559368133544922
-7.062349319458008 1.5570945739746094 4.852226257324219
1.5742721557617188 8.544479370117188 0.62603759765625
23.8385009765625 -6.328737258911133 -7.715229034423828
0.8879928588867188 14.415115356445312 0.066009521484375
-8.397090911865234 -4.053056716918945 8.456405639648438
20.177776336669922 -8.935157775878906 -0.471405029296875
-7.908267974853516 -4.574615478515625 8.534408569335938
27.24231719970703 19.05392074584961 -16.591873168945312
13.963611602783203 -5.595468521118164 -6.33222770690918
24.845352172851562 22.69097900390625 -23.129671096801758
face_pose_estimate_single_hopenet time: 1.0039 seconds
face_pose_estimate_single_shuffledhopenet time: 0.4170 seconds
face_pose_estimate_single_hopenetlite time: 0.3979 seconds
----------------
batch_pose_estimate_hopenet time: 0.5564 seconds
batch_pose_estimate_shuffledhopenet time: 0.1433 seconds
batch_pose_estimate_hopenetlite time: 0.1354 seconds
16.85895 4.3864007 -14.910521
12.910108 4.9568367 -17.696342
14.657368 3.3150046 -16.276155
7.427281 -5.5885296 -20.4827
9.6222105 -14.666496 -24.1662
7.5376196 -9.090814 -23.099
-24.916548 -6.491705 3.8903186
-26.697842 -7.6093497 2.9928446
-22.337896 -5.424592 5.2471504
-6.4662757 0.7829325 3.6133397
-8.892705 -4.0846605 2.4946926
-7.062358 1.5571051 4.852224
1.5742786 8.54448 0.6260239
-1.189327 13.24428 -2.952901
0.8879808 14.415126 0.06601605
-8.397098 -4.0530615 8.456407
-8.810427 -6.0657454 5.2215085
-7.9082737 -4.5746093 8.5344095
27.24231 19.053915 -16.591879
19.81821 23.090067 -15.712087
24.845333 22.69098 -23.129665
---------------
face_pose_estimate_batch_hopenet time: 0.5312 seconds
face_pose_estimate_batch_shuffledhopenet time: 0.1273 seconds
face_pose_estimate_batch_hopenetlite time: 0.1283 seconds
16.858948 4.3863983 -14.910522
12.91011 4.9568253 -17.696335
14.657379 3.31501 -16.276161
7.4272842 -5.588539 -20.482697
9.622215 -14.666504 -24.166199
7.537613 -9.090805 -23.099007
-24.91655 -6.491699 3.8903198
-26.69786 -7.6093445 2.9928436
-22.337898 -5.424576 5.2471313
-6.466278 0.7829361 3.6133347
-8.892715 -4.084671 2.4946976
-7.0623474 1.557106 4.852234
1.574295 8.544479 0.62602234
-1.1893158 13.244293 -2.952896
0.88797 14.415115 0.06602478
-8.397102 -4.0530624 8.456406
-8.810425 -6.065735 5.221527
-7.908287 -4.5746155 8.534424
27.242317 19.05391 -16.591888
19.818207 23.090073 -15.712082
24.845352 22.690979 -23.12967
    """



if __name__ == "__main__":
    # face_pose_pipeline('resources/pictures/input/1-1.jpeg', 'resources/pictures/output/test_pr_hopenet_1-1.jpeg')
    # face_align()
    # face_point()
    # recognition()
    # compare_hopenet()
    # align()
    # batch_hopenet1()
    from algorithm.face_recognition import process_directory

    process_directory()
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
