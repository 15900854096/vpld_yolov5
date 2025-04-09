# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Image augmentation functions
"""

import math
import random
import time
import copy

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from utils.general import LOGGER, check_version, colorstr, resample_segments, segment2box, xywhn2xyxy
from utils.metrics import bbox_ioa

IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation


class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self, size=640):
        self.transform = None
        prefix = colorstr('albumentations: ')
        try:
            import albumentations as A
            check_version(A.__version__, '1.0.3', hard=True)  # version requirement

            T = [
                A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0)]  # transforms
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

            LOGGER.info(prefix + ', '.join(f'{x}'.replace('always_apply=False, ', '') for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f'{prefix}{e}')

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        return im, labels


def normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace=False):
    # Denormalize RGB images x per ImageNet stats in BCHW format, i.e. = (x - mean) / std
    return TF.normalize(x, mean, std, inplace=inplace)


def denormalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    # Denormalize RGB images x per ImageNet stats in BCHW format, i.e. = x * std + mean
    for i in range(3):
        x[:, i] = x[:, i] * std[i] + mean[i]
    return x


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    np.random.seed(time.time_ns()%(2**32 - 1))
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


def hist_equalize(im, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'im' with im.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def replicate(im, labels):
    # Replicate labels
    h, w = im.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]  # im4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return im, labels


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def random_perspective(im,
                       targets=(),
                       segments=(),
                       degrees=10,
                       translate=.1,
                       scale=.1,
                       shear=10,
                       perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments) and len(segments) == n
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets


def copy_paste(im, labels, segments, p=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (1, 1, 1), cv2.FILLED)

        result = cv2.flip(im, 1)  # augment segments (flip left-right)
        i = cv2.flip(im_new, 1).astype(bool)
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

    return im, labels, segments


def cutout(im, labels, p=0.5):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    if random.random() < p:
        h, w = im.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        for s in scales:
            mask_h = random.randint(1, int(h * s))  # create random masks
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

            # return unobscured labels
            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, xywhn2xyxy(labels[:, 1:5], w, h))  # intersection over area
                labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def mixup(im, labels, im2, labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


def classify_albumentations(
        augment=True,
        size=224,
        scale=(0.08, 1.0),
        ratio=(0.75, 1.0 / 0.75),  # 0.75, 1.33
        hflip=0.5,
        vflip=0.0,
        jitter=0.4,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        auto_aug=False):
    # YOLOv5 classification Albumentations (optional, only used if package is installed)
    prefix = colorstr('albumentations: ')
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        check_version(A.__version__, '1.0.3', hard=True)  # version requirement
        if augment:  # Resize and crop
            T = [A.RandomResizedCrop(height=size, width=size, scale=scale, ratio=ratio)]
            if auto_aug:
                # TODO: implement AugMix, AutoAug & RandAug in albumentation
                LOGGER.info(f'{prefix}auto augmentations are currently not supported')
            else:
                if hflip > 0:
                    T += [A.HorizontalFlip(p=hflip)]
                if vflip > 0:
                    T += [A.VerticalFlip(p=vflip)]
                if jitter > 0:
                    color_jitter = (float(jitter),) * 3  # repeat value for brightness, contrast, satuaration, 0 hue
                    T += [A.ColorJitter(*color_jitter, 0)]
        else:  # Use fixed crop for eval set (reproducibility)
            T = [A.SmallestMaxSize(max_size=size), A.CenterCrop(height=size, width=size)]
        T += [A.Normalize(mean=mean, std=std), ToTensorV2()]  # Normalize and convert to Tensor
        LOGGER.info(prefix + ', '.join(f'{x}'.replace('always_apply=False, ', '') for x in T if x.p))
        return A.Compose(T)

    except ImportError:  # package not installed, skip
        LOGGER.warning(f'{prefix}⚠️ not found, install with `pip install albumentations` (recommended)')
    except Exception as e:
        LOGGER.info(f'{prefix}{e}')


def classify_transforms(size=224):
    # Transforms to apply if albumentations not installed
    assert isinstance(size, int), f'ERROR: classify_transforms size {size} must be integer, not (list, tuple)'
    # T.Compose([T.ToTensor(), T.Resize(size), T.CenterCrop(size), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose([CenterCrop(size), ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class LetterBox:
    # YOLOv5 LetterBox class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, size=(640, 640), auto=False, stride=32):
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto  # pass max size integer, automatically solve for short side using stride
        self.stride = stride  # used with auto

    def __call__(self, im):  # im = np.array HWC
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)  # ratio of new/old
        h, w = round(imh * r), round(imw * r)  # resized image
        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else self.h, self.w
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)
        im_out = np.full((self.h, self.w, 3), 114, dtype=im.dtype)
        im_out[top:top + h, left:left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im_out


class CenterCrop:
    # YOLOv5 CenterCrop class for image preprocessing, i.e. T.Compose([CenterCrop(size), ToTensor()])
    def __init__(self, size=640):
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):  # im = np.array HWC
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(im[top:top + m, left:left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)


class ToTensor:
    # YOLOv5 ToTensor class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, half=False):
        super().__init__()
        self.half = half

    def __call__(self, im):  # im = np.array HWC in BGR order
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im





#add by xuqing
def gt_flipud(t):
    c,p0,p1,p2,p3 =  np.split(t , [1, 3, 5, 7], axis=1)
    p0[:,1]=1-p0[:,1]
    p1[:,1]=1-p1[:,1]
    p2[:,1]=1-p2[:,1]
    p3[:,1]=1-p3[:,1]
    t=np.concatenate([c,p1,p0,p3,p2],axis=1)
    return t

def gt_fliplr(t):
    c,p0,p1,p2,p3 =  np.split(t , [1, 3, 5, 7], axis=1)
    p0[:,0]=1-p0[:,0]
    p1[:,0]=1-p1[:,0]
    p2[:,0]=1-p2[:,0]
    p3[:,0]=1-p3[:,0]
    t=np.concatenate([c,p1,p0,p3,p2],axis=1)
    return t

def gt_rotate90degree(t, direction):
    c,p0,p1,p2,p3 =  np.split(t , [1, 3, 5, 7], axis=1)
    if (cv2.ROTATE_90_CLOCKWISE==direction):
        temx, temy = 1-p0[:,1], p0[:,0]
        p0[:,0], p0[:,1] = copy.deepcopy(temx), copy.deepcopy(temy)
        
        temx, temy =  1-p1[:,1], p1[:,0]
        p1[:,0], p1[:,1] = copy.deepcopy(temx), copy.deepcopy(temy)

        temx, temy =  1-p2[:,1], p2[:,0]
        p2[:,0], p2[:,1] = copy.deepcopy(temx), copy.deepcopy(temy)

        temx, temy =  1-p3[:,1], p3[:,0]
        p3[:,0], p3[:,1] = copy.deepcopy(temx), copy.deepcopy(temy)
    elif(cv2.ROTATE_90_COUNTERCLOCKWISE==direction):
        temx, temy = p0[:,1], 1-p0[:,0]
        p0[:,0], p0[:,1] = copy.deepcopy(temx), copy.deepcopy(temy)
        
        temx, temy = p1[:,1], 1-p1[:,0]
        p1[:,0], p1[:,1] = copy.deepcopy(temx), copy.deepcopy(temy)

        temx, temy = p2[:,1], 1-p2[:,0]
        p2[:,0], p2[:,1] = copy.deepcopy(temx), copy.deepcopy(temy)

        temx, temy = p3[:,1], 1-p3[:,0]
        p3[:,0], p3[:,1] = copy.deepcopy(temx), copy.deepcopy(temy)
    else:
        print("direction error")
        sys.exit()
    t=np.concatenate([c,p0,p1,p2,p3],axis=1)
    return t


def gt_rotate(t,oW,oH,nW, nH,M):
    c,p0,p1,p2,p3 =  np.split(t , [1, 3, 5, 7], axis=1)
    
    tmp0 = np.zeros_like(p0)
    tmp0[:,0]=(M[0,0]*p0[:,0]*oW+M[0,1]*p0[:,1]*oH+1*M[0,2])/nW
    tmp0[:,1]=(M[1,0]*p0[:,0]*oW+M[1,1]*p0[:,1]*oH+1*M[1,2])/nH
    p0=tmp0
    
    tmp1 = np.zeros_like(p1)  
    tmp1[:,0]=(M[0,0]*p1[:,0]*oW+M[0,1]*p1[:,1]*oH+1*M[0,2])/nW
    tmp1[:,1]=(M[1,0]*p1[:,0]*oW+M[1,1]*p1[:,1]*oH+1*M[1,2])/nH
    p1=tmp1

    tmp2 = np.zeros_like(p2)
    tmp2[:,0]=(M[0,0]*p2[:,0]*oW+M[0,1]*p2[:,1]*oH+1*M[0,2])/nW
    tmp2[:,1]=(M[1,0]*p2[:,0]*oW+M[1,1]*p2[:,1]*oH+1*M[1,2])/nH
    p2=tmp2

    tmp3 = np.zeros_like(p3)
    tmp3[:,0]=(M[0,0]*p3[:,0]*oW+M[0,1]*p3[:,1]*oH+1*M[0,2])/nW
    tmp3[:,1]=(M[1,0]*p3[:,0]*oW+M[1,1]*p3[:,1]*oH+1*M[1,2])/nH
    p3=tmp3

    return np.concatenate([c,p0,p1,p2,p3],axis=1)

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    if 0: 
        nW = w#int((h * sin) + (w * cos))
        nH = h#int((h * cos) + (w * sin))
    else:
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH)),nW, nH,M    


def is_gt_out_img(lot, min_value, max_value):
    if(lot[1]<min_value or lot[2]<min_value or lot[3]<min_value or lot[4]<min_value):
        return False
    if(lot[1]>max_value or lot[2]>max_value or lot[3]>max_value or lot[4]>max_value):
        return False    
    return True

def image_gt_data_resize_all(img, gt):
    np.random.seed(time.time_ns()%(2**32 - 1))
    H,W,C = img.shape
    imgsize = H
    rate = np.random.uniform(0.9, 1.1, 1)
    if(rate>1):#图像贴在大图上，再resize，相当于边界填充  相当于库位变小
        masksize = int(imgsize*rate)
        while(masksize == imgsize):
            rate = np.random.uniform(1, 1.1, 1)
            masksize = int(imgsize*rate)
        mask = np.random.randint(0, 255, [masksize,masksize,3],dtype=np.uint8)
        start_row = np.random.randint(0, masksize-imgsize)
        start_col = np.random.randint(0, masksize-imgsize)
        mask[start_row:start_row+imgsize,start_col:start_col+imgsize,:] = img
        img_res = cv2.resize(mask, (imgsize, imgsize))
        if(gt.shape[0]>0):
            gt_res = gt * np.array([1,imgsize,imgsize,imgsize,imgsize,imgsize,imgsize,imgsize,imgsize])
            gt_res = gt_res + np.array([0,start_col, start_row, start_col, start_row, start_col, start_row, start_col, start_row])
            gt_res = gt_res / np.array([1,masksize,masksize,masksize,masksize,masksize,masksize,masksize,masksize])
        else:
            gt_res = gt
        return img_res,gt_res
    else:#图像放大再裁剪其中一部分  相当于库位变大
        masksize = int(imgsize/rate)
        while(masksize == imgsize):
            rate = np.random.uniform(0.9, 1, 1)
            masksize = int(imgsize/rate)
        img = cv2.resize(img, (masksize, masksize))
        start_row = np.random.randint(0, masksize-imgsize)
        start_col = np.random.randint(0, masksize-imgsize)
        img_res = img[start_row:start_row+imgsize,start_col:start_col+imgsize,:]
        if(gt.shape[0]>0):
            gt_res = gt * np.array([1,masksize,masksize,masksize,masksize,masksize,masksize,masksize,masksize])
            gt_res = gt_res - np.array([0,start_col, start_row, start_col, start_row, start_col, start_row, start_col, start_row])
            select = [is_gt_out_img(lot, 10, imgsize-10) for lot in gt_res]
            gt_res = gt_res[select]
            if(gt_res.shape[0]>0):
                gt_res = gt_res / np.array([1,imgsize,imgsize,imgsize,imgsize,imgsize,imgsize,imgsize,imgsize])
        else:
            gt_res = gt
        return img_res, gt_res
    
    
def sunlight(img):
    random.seed(time.time_ns()%(2**32 - 1))
    #获取图像行和列
    rows, cols = img.shape[:2]
    #设置中心点
    centerX = random.randint(0,rows)
    centerY =  random.randint(0,cols)
    #print(centerX,centerY) 
    radius = random.randint(30,60)
    strength = random.randint(50,150)
    
    img = img.astype(np.float32)
    
    imgpad = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    imgpad = cv2.circle(imgpad, (centerX, centerY), radius, (strength, strength, strength), -1)

    imgpad = cv2.GaussianBlur(imgpad,(21,21),20,20)
    img = img+np.repeat(imgpad,3,axis=1).reshape(img.shape)

    img[img>255]=255
    img[img<0]=0
   
    return img.astype(np.uint8)


def rain(img):
    random.seed(time.time_ns()%(2**32 - 1))
    value = random.randint(10,500)
    length = random.randint(10,50) # 对角矩阵大小，表示雨滴的长度
    angle = random.randint(-30,30) # 倾斜的角度，逆时针为正
    w = random.randint(1,5)//2*2+1 # 雨滴大小
    beta = np.random.uniform(0.1, 0.8)
    #print(value,length,angle,w,beta)
    
    #noise
    noise = np.random.uniform(0, 256, img.shape[0:2])
    # 控制噪声水平，取浮点数，只保留最大的一部分作为噪声
    v = value * 0.01
    noise[np.where(noise < (256 - v))] = 0
    # 噪声做初次模糊
    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])
    noise = cv2.filter2D(noise, -1, k)
    
    #blurred
    #这里由于对角阵自带45度的倾斜，逆时针为正，所以加了-45度的误差，保证开始为正
    trans = cv2.getRotationMatrix2D((length/2, length/2), angle-45, 1-length/100.0)  
    dig = np.diag(np.ones(length))   #生成对焦矩阵
    k = cv2.warpAffine(dig, trans, (length, length))  #生成模糊核
    k = cv2.GaussianBlur(k,(w,w),0)    #高斯模糊这个旋转后的对角核，使得雨有宽度
    blurred = cv2.filter2D(noise, -1, k)    #用刚刚得到的旋转后的核，进行滤波
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX) #转换到0-255区间
    blurred = np.array(blurred, dtype=np.uint8)
    
    rain = np.expand_dims(blurred,2)
    rain_effect = np.concatenate((img,rain),axis=2)  #add alpha channel
    rain_result = img.copy()    #拷贝一个掩膜
    rain = np.array(rain,dtype=np.float32)     #数据类型变为浮点数，后面要叠加，防止数组越界要用32位
    rain_result[:,:,0] = rain_result[:,:,0] * (255-rain[:,:,0])/255.0 + beta*rain[:,:,0]
    rain_result[:,:,1] = rain_result[:,:,1] * (255-rain[:,:,0])/255 + beta*rain[:,:,0] 
    rain_result[:,:,2] = rain_result[:,:,2] * (255-rain[:,:,0])/255 + beta*rain[:,:,0]
    
    return rain_result


def AddGaussianNoise(img):
    random.seed(time.time_ns()%(2**32 - 1))
    mean = random.randint(0,20)
    var = random.randint(5,15)
    #print(mean,var)
    img = img.astype(np.float32)
    noise =  np.random.normal(mean,var,img.shape)#产生相同shape的噪声数组
    img=img+noise  
    img[img>255]=255
    img[img<0]=0   
    return img.astype(np.uint8)

def AddPepperSaltNoise(img):
    np.random.seed(time.time_ns()%(2**32 - 1))
    percent = np.random.uniform(0.0001, 0.05)
    img = img.astype(np.float32)

    all_num = int(percent*img.shape[0]*img.shape[1])
    black_num = int(all_num*np.random.uniform(0.1, 0.9))

    xselect = np.random.randint(0,img.shape[0]-1,all_num)
    yselect = np.random.randint(0,img.shape[1]-1,all_num)
    cselect = np.random.randint(0,img.shape[2]-1,all_num)
    img[xselect,yselect,cselect]=0

    temp = np.random.randint(0,all_num-1,black_num)
    img[xselect[temp],yselect[temp],cselect[temp]]=255        
    return img.astype(np.uint8)


def value_range(x,minvalue,maxvalue):
    if x<minvalue:
        return minvalue
    if x>maxvalue:
        return maxvalue
    return x

def cutblock(img,labels):
    random.seed(time.time_ns()%(2**32 - 1))
    bigrange = 10
    for label in labels:
        probility = random.random()
        if probility < 0.5:
            continue
        center_x = int(label[1]*img.shape[1]) # H W C
        center_y = int(label[2]*img.shape[0])
        lefttop_x = value_range(center_x - bigrange, 0, img.shape[1]-1)
        lefttop_y = value_range(center_y - bigrange, 0, img.shape[0]-1)
        rightbottom_x = value_range(center_x + bigrange, 0, img.shape[1]-1)
        rightbottom_y = value_range(center_y + bigrange, 0, img.shape[0]-1)
        
        length = random.randint(10,20)
        x = random.randint(lefttop_x, rightbottom_x)
        y = random.randint(lefttop_y, rightbottom_y)
        lfx = value_range(x-length, 0, img.shape[1]-1)
        lfy = value_range(y-length, 0, img.shape[0]-1)
        rbx = value_range(x+length, 0, img.shape[1]-1)
        rby = value_range(y+length, 0, img.shape[0]-1)
        img[lfy:rby,lfx:rbx,:] = 0
        if probility < 0.25:
            x = random.randint(0, lefttop_x)
            y = random.randint(0, lefttop_y)
            lfx = value_range(x-length, 0, img.shape[1]-1)
            lfy = value_range(y-length, 0, img.shape[0]-1)
            rbx = value_range(x+length, 0, img.shape[1]-1)
            rby = value_range(y+length, 0, img.shape[0]-1)
            img[lfy:rby,lfx:rbx,:] = 0
        else:
            x = random.randint(rightbottom_x, img.shape[1]-1)
            y = random.randint(rightbottom_y, img.shape[0]-1)
            lfx = value_range(x-length, 0, img.shape[1]-1)
            lfy = value_range(y-length, 0, img.shape[0]-1)
            rbx = value_range(x+length, 0, img.shape[1]-1)
            rby = value_range(y+length, 0, img.shape[0]-1)
            img[lfy:rby,lfx:rbx,:] = 0
        
    return    