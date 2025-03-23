import torch
from torch.nn.functional import interpolate
from torchvision.transforms import functional as F
from torchvision.ops.boxes import batched_nms
from PIL import Image
import numpy as np
import os
import math

# OpenCV is optional, but required if using numpy arrays instead of PIL
try:
    import cv2
except:
    pass


def fixed_batch_process(im_data, model):
    """保持原有分批逻辑，适配单输出模型"""
    batch_size = 512
    out = []
    for i in range(0, len(im_data), batch_size):
        batch = im_data[i:(i + batch_size)]
        out.append(model(batch))
    return torch.cat(out, dim=0)  # 假设现在返回单张量


def detect_landmarks(imgs, onet, device, threshold=0.7):
    # Convert input to tensor and resize to 48x48
    if isinstance(imgs, (np.ndarray, torch.Tensor)):
        if isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs.copy()).to(device)
        elif imgs.device != device:
            imgs = imgs.to(device)

        if len(imgs.shape) == 3:
            imgs = imgs.unsqueeze(0)

        # Resize if not 48x48
        if imgs.shape[-2:] != (48, 48):
            imgs = imgs.permute(0, 3, 1, 2) if imgs.shape[1] != 3 else imgs
            imgs = interpolate(imgs, size=(48, 48), mode='bilinear', align_corners=False)
    else:
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]

        resized_imgs = []
        for img in imgs:
            if isinstance(img, Image.Image):
                img = img.resize((48, 48), Image.BILINEAR)
                img = F.to_tensor(img)
            elif isinstance(img, np.ndarray):
                img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_LINEAR)
                img = torch.from_numpy(img).permute(2, 0, 1).float()
            resized_imgs.append(img)

        imgs = torch.stack(resized_imgs).to(device)

    # Ensure channel-first format
    if imgs.shape[1] != 3:
        imgs = imgs.permute(0, 3, 1, 2)

    # Normalize
    imgs = (imgs - 127.5) * 0.0078125

    with torch.no_grad():
        points = fixed_batch_process(imgs, onet)
    w_i = imgs.shape[-1]
    h_i = imgs.shape[-2]
    points_x = w_i.repeat(5, 1) * points[:5, :] + boxes[:, 0].repeat(5, 1) - 1
    points_y = h_i.repeat(5, 1) * points[5:10, :] + boxes[:, 1].repeat(5, 1) - 1
    points = torch.stack((points_x, points_y)).permute(2, 1, 0)


def bbreg(boundingbox, reg):
    if reg.shape[1] == 1:
        reg = torch.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, :4] = torch.stack([b1, b2, b3, b4]).permute(1, 0)

    return boundingbox


def generateBoundingBox(reg, probs, scale, thresh):
    stride = 2
    cellsize = 12

    reg = reg.permute(1, 0, 2, 3)

    mask = probs >= thresh
    mask_inds = mask.nonzero()
    image_inds = mask_inds[:, 0]
    score = probs[mask]
    reg = reg[:, mask].permute(1, 0)
    bb = mask_inds[:, 1:].type(reg.dtype).flip(1)
    q1 = ((stride * bb + 1) / scale).floor()
    q2 = ((stride * bb + cellsize - 1 + 1) / scale).floor()
    boundingbox = torch.cat([q1, q2, score.unsqueeze(1), reg], dim=1)
    return boundingbox, image_inds


def nms_numpy(boxes, scores, threshold, method):
    if boxes.size == 0:
        return np.empty((0, 3))

    x1 = boxes[:, 0].copy()
    y1 = boxes[:, 1].copy()
    x2 = boxes[:, 2].copy()
    y2 = boxes[:, 3].copy()
    s = scores
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]

        xx1 = np.maximum(x1[i], x1[idx]).copy()
        yy1 = np.maximum(y1[i], y1[idx]).copy()
        xx2 = np.minimum(x2[i], x2[idx]).copy()
        yy2 = np.minimum(y2[i], y2[idx]).copy()

        w = np.maximum(0.0, xx2 - xx1 + 1).copy()
        h = np.maximum(0.0, yy2 - yy1 + 1).copy()

        inter = w * h
        if method == 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o <= threshold)]

    pick = pick[:counter].copy()
    return pick


def batched_nms_numpy(boxes, scores, idxs, threshold, method):
    device = boxes.device
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    boxes_for_nms = boxes_for_nms.cpu().numpy()
    scores = scores.cpu().numpy()
    keep = nms_numpy(boxes_for_nms, scores, threshold, method)
    return torch.as_tensor(keep, dtype=torch.long, device=device)


def pad(boxes, w, h):
    boxes = boxes.trunc().int().cpu().numpy()
    x = boxes[:, 0]
    y = boxes[:, 1]
    ex = boxes[:, 2]
    ey = boxes[:, 3]

    x[x < 1] = 1
    y[y < 1] = 1
    ex[ex > w] = w
    ey[ey > h] = h

    return y, ey, x, ex


def rerec(bboxA):
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]

    l = torch.max(w, h)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, :2] + l.repeat(2, 1).permute(1, 0)

    return bboxA


def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return im_data


def crop_resize(img, box, image_size):
    if isinstance(img, np.ndarray):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = cv2.resize(
            img,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA
        ).copy()
    elif isinstance(img, torch.Tensor):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = imresample(
            img.permute(2, 0, 1).unsqueeze(0).float(),
            (image_size, image_size)
        ).byte().squeeze(0).permute(1, 2, 0)
    else:
        out = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
    return out


def save_img(img, path):
    if isinstance(img, np.ndarray):
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        img.save(path)


def get_size(img):
    if isinstance(img, (np.ndarray, torch.Tensor)):
        return img.shape[1::-1]
    else:
        return img.size


def extract_face(img, box, image_size=160, margin=0, save_path=None):
    """Extract face + margin from PIL Image given bounding box.

    Arguments:
        img {PIL.Image} -- A PIL Image.
        box {numpy.ndarray} -- Four-element bounding box.
        image_size {int} -- Output image size in pixels. The image will be square.
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image.
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size.
        save_path {str} -- Save path for extracted face image. (default: {None})

    Returns:
        torch.tensor -- tensor representing the extracted face.
    """
    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin),
    ]
    raw_image_size = get_size(img)
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, raw_image_size[0])),
        int(min(box[3] + margin[1] / 2, raw_image_size[1])),
    ]

    face = crop_resize(img, box, image_size)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) + "/", exist_ok=True)
        save_img(face, save_path)

    face = F.to_tensor(np.float32(face))

    return face
