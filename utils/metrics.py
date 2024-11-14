# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Model validation metrics
"""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import TryExcept, threaded

def fitness(x):
    # Model fitness as a weighted combination of metrics
    #x=[P, R, mAP@0.5, mAP@0.95, mAP@0.5:0.95, loss_box, loss_obj, loss_cls]
    w = [0.0, 0.0, 0.1, 0.9, 0.1]  # weights for [P, R, mAP@0.5,mAP@0.95, mAP@0.5:0.95]
    return (x[:, :5] * w).sum(1)


def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16, prefix=''):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / f'{prefix}PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / f'{prefix}F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / f'{prefix}P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / f'{prefix}R_curve.png', names, ylabel='Recall')

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        if detections is None:
            gt_classes = labels.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        detections = detections[detections[:, 8] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 9].int()
        #iou = box_iou(labels[:, 1:], detections[:, :4])
        iou = box_iou_poly(labels[:, 1:], detections[:, :8])
        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background

    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    @TryExcept('WARNING ⚠️ ConfusionMatrix plot failure')
    def plot(self, normalize=True, save_dir='', names=()):
        import seaborn as sn

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (names + ['background']) if labels else 'auto'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(array,
                       ax=ax,
                       annot=nc < 30,
                       annot_kws={
                           'size': 8},
                       cmap='Blues',
                       fmt='.2f',
                       square=True,
                       vmin=0.0,
                       xticklabels=ticklabels,
                       yticklabels=ticklabels).set_facecolor((1, 1, 1))
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title('Confusion Matrix')
        fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        plt.close(fig)

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_ioa(box1, box2, eps=1e-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def wh_iou(wh1, wh2, eps=1e-7):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter + eps)  # iou = inter / (area1 + area2 - inter)


# Plots ----------------------------------------------------------------------------------------------------------------


@threaded
def plot_pr_curve(px, py, ap, save_dir=Path('pr_curve.png'), names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title('Precision-Recall Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


@threaded
def plot_mc_curve(px, py, save_dir=Path('mc_curve.png'), names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title(f'{ylabel}-Confidence Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)

import shapely
import sys
from shapely.geometry import Polygon, MultiPoint
def bbox_iou_eval(box1, box2):
    box1 = np.array(box1.cpu()).reshape(4, 2)  # 四边形二维坐标表示
    # python四边形对象，会自动计算四个点，并将四个点重新排列成
    # 左上，左下，右下，右上，左上（没错左上排了两遍）
    poly1 = Polygon(box1).convex_hull
    box2 = np.array(box2.cpu()).reshape(4, 2)
    poly2 = Polygon(box2).convex_hull
    if not poly1.intersects(poly2):  # 如果两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # 相交面积
            iou = float(inter_area) / (poly1.area + poly2.area - inter_area)
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou

#统计自车在库位内部的面积占自车面积的比例，即自车有多少占比已经入库了
def bbox_iou_cat_in_lot(poly1, poly2):
    #box1 must be car Polygon
    #box2 must be lot Polygon
    if not poly1.intersects(poly2):  # 如果两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # 相交面积
            iou = float(inter_area) / poly1.area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou

def box_iou_poly(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    #predn:   x1 y1 x2 y2 x3 y3 x4 y4 all base_ori
    #labelsn: x1 y1 x2 y2 x3 y3 x4 y4 base_ori&depth_ok
    N = box1.shape[0]
    M = box2.shape[0]
    tmp1=box1 #torch.cat((box1,box1[:,0:2]+box1[:,4:6]-box1[:,2:4]),1)
    tmp2=box2 #torch.cat((box2,box2[:,0:2]+box2[:,4:6]-box2[:,2:4]),1)
    
    ioumat=torch.zeros((N,M),device = box1.device)
    for i in range(N):
        for j in range(M):
            ioumat[i][j]=bbox_iou_eval(tmp1[i], tmp2[j])
    return ioumat        

class Cal_R_Matrix:
    def __init__(self,imgsz,device):
        self.device = device
        self.A_x_err = []
        self.A_y_err = []
        self.B_x_err = []
        self.B_y_err = []
        self.AD_theta_err = []
        self.BC_theta_err = []
        self.search_park = []
        self.cnt = 0
        self.imgsz = imgsz
        self.polycar = np.array([270/640.0, 193/640.0, 370/640.0, 193/640.0, 370/640.0, 445/640.0, 270/640.0, 445/640.0]).reshape(4, 2)  # 四边形二维坐标表示
        self.polycar = Polygon(self.polycar).convex_hull

    def reset(self):
        self.A_x_err = []
        self.A_y_err = []
        self.B_x_err = []
        self.B_y_err = []
        self.AD_theta_err = []
        self.BC_theta_err = []
        self.cnt = 0
    
    def get_real_theta(self, delta):
        from utils.general import PI
        delta = abs(delta)
        return delta if delta<PI else 2*PI-delta
        # if delta < PI and delta > -PI:
        #     return delta
        # elif delta > PI:
        #     return 2*PI-delta
        # elif delta < -PI:
        #     return 2*PI+delta
        
    
    def update(self, detections, labels, sz, iou_thres=0.9):
        boxes_labels = labels[:, 1:] / sz 
        boxes_detections = detections[:, :8] / sz
        
        N = boxes_labels.shape[0]
        M = boxes_detections.shape[0]
        
        for i in range(N):
            for j in range(M):
                iou = bbox_iou_eval(boxes_labels[i], boxes_detections[j])
                is_parking_lot = not self.polycar.disjoint(Polygon(np.array(boxes_labels[i].cpu()).reshape(4,2)).convex_hull) # Returns True if A and B do not share any point in space. 
                if (    (iou > iou_thres) ): 
                    #  0  1  2  3  4  5  6  7
                    # x1 y1 x2 y2 x3 y3 x4 y4
                    label_AD_theta = torch.atan2(boxes_labels[i][7]-boxes_labels[i][1], boxes_labels[i][6]-boxes_labels[i][0])
                    label_BC_theta = torch.atan2(boxes_labels[i][5]-boxes_labels[i][3], boxes_labels[i][4]-boxes_labels[i][2])
                    
                    detection_AD_theta = torch.atan2(boxes_detections[j][7]-boxes_detections[j][1], boxes_detections[j][6]-boxes_detections[j][0])
                    detection_BC_theta = torch.atan2(boxes_detections[j][5]-boxes_detections[j][3], boxes_detections[j][4]-boxes_detections[j][2])
                    
                    self.A_x_err.append( (boxes_labels[i][0] - boxes_detections[j][0]) * self.imgsz )
                    self.A_y_err.append( (boxes_labels[i][1] - boxes_detections[j][1]) * self.imgsz )
                    self.B_x_err.append( (boxes_labels[i][2] - boxes_detections[j][2]) * self.imgsz )
                    self.B_y_err.append( (boxes_labels[i][3] - boxes_detections[j][3]) * self.imgsz )
                    self.AD_theta_err.append( self.get_real_theta(label_AD_theta - detection_AD_theta) )
                    self.BC_theta_err.append( self.get_real_theta(label_BC_theta - detection_BC_theta) )
                    self.search_park.append( is_parking_lot )
                    self.cnt+=1

    def get_result(self):
        if(len(self.A_x_err)==0):
            return
            
        A_x_err = np.array(torch.stack(self.A_x_err).cpu())
        A_y_err = np.array(torch.stack(self.A_y_err).cpu())
        B_x_err = np.array(torch.stack(self.B_x_err).cpu())
        B_y_err = np.array(torch.stack(self.B_y_err).cpu())
        AD_theta_err = np.array(torch.stack(self.AD_theta_err).cpu())
        BC_theta_err = np.array(torch.stack(self.BC_theta_err).cpu())
        search_park = np.array(self.search_park)

        flag = False
        cnt = search_park[search_park==flag].shape[0]
        if(cnt):
            print("searching cnt: ", cnt)
            print("searching A_x_err: ",      torch.mean(torch.abs(torch.tensor(A_x_err[search_park==flag]))),      torch.mean(torch.pow(torch.tensor(A_x_err[search_park==flag]),2)))
            print("searching A_y_err: ",      torch.mean(torch.abs(torch.tensor(A_y_err[search_park==flag]))),      torch.mean(torch.pow(torch.tensor(A_y_err[search_park==flag]),2)))
            print("searching B_x_err: ",      torch.mean(torch.abs(torch.tensor(B_x_err[search_park==flag]))),      torch.mean(torch.pow(torch.tensor(B_x_err[search_park==flag]),2)))
            print("searching B_y_err: ",      torch.mean(torch.abs(torch.tensor(B_y_err[search_park==flag]))),      torch.mean(torch.pow(torch.tensor(B_y_err[search_park==flag]),2)))
            print("searching AD_theta_err: ", torch.mean(torch.abs(torch.tensor(AD_theta_err[search_park==flag]))), torch.mean(torch.pow(torch.tensor(AD_theta_err[search_park==flag]),2)))
            print("searching BC_theta_err: ", torch.mean(torch.abs(torch.tensor(BC_theta_err[search_park==flag]))), torch.mean(torch.pow(torch.tensor(BC_theta_err[search_park==flag]),2)))
            print("\n")
            
        flag = True
        cnt = search_park[search_park==flag].shape[0] 
        if(cnt):
            print("parking cnt: ", cnt)
            print("parking A_x_err: ",      torch.mean(torch.abs(torch.tensor(A_x_err[search_park==flag]))),      torch.mean(torch.pow(torch.tensor(A_x_err[search_park==flag]),2)))
            print("parking A_y_err: ",      torch.mean(torch.abs(torch.tensor(A_y_err[search_park==flag]))),      torch.mean(torch.pow(torch.tensor(A_y_err[search_park==flag]),2)))
            print("parking B_x_err: ",      torch.mean(torch.abs(torch.tensor(B_x_err[search_park==flag]))),      torch.mean(torch.pow(torch.tensor(B_x_err[search_park==flag]),2)))
            print("parking B_y_err: ",      torch.mean(torch.abs(torch.tensor(B_y_err[search_park==flag]))),      torch.mean(torch.pow(torch.tensor(B_y_err[search_park==flag]),2)))
            print("parking AD_theta_err: ", torch.mean(torch.abs(torch.tensor(AD_theta_err[search_park==flag]))), torch.mean(torch.pow(torch.tensor(AD_theta_err[search_park==flag]),2)))
            print("parking BC_theta_err: ", torch.mean(torch.abs(torch.tensor(BC_theta_err[search_park==flag]))), torch.mean(torch.pow(torch.tensor(BC_theta_err[search_park==flag]),2)))
            print("\n")