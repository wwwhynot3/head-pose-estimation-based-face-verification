import cv2
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import algorithm.face_pose_estimation.utils as utils
import algorithm.model.hopenet as hopenet


def test_hopenet():
    cudnn.enabled = True

    # 1. 加载Hopenet模型
    model = hopenet.HopeNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    saved_state_dict = torch.load('resources/model/face_pose_estimation.pkl', map_location=torch.device('cpu'))
    model.load_state_dict(saved_state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 动态分配设备
    model = model.to(device)  # 替换model.cuda()
    model.eval()
    print('Hopenet model loaded success')
    # 2. 图像预处理流程[1](@ref)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. 读取并处理图像
    img = cv2.imread("resources/pictures/input/1-2.jpeg")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
    input_img = transform(img_rgb).unsqueeze(0)  # 增加batch维度

    # 4. 推理预测
    with torch.no_grad():
        images = input_img.to(device)  # 替换Variable(input_img.cuda())
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
    print(yaw_deg, pitch_deg, roll_deg)
    # 6. 绘制姿态轴并保存
    img = utils.draw_axis(img, yaw_deg, pitch_deg, roll_deg)
    res = cv2.imwrite("resources/pictures/output/1_2_out.jpeg", img)
    print(f'write success: {res}')
    print('done')
