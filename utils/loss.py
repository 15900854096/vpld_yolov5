# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
import sys
import time
import yaml
import os
import random
import copy
import pandas as pd
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPoint
from time import perf_counter_ns
from utils.metrics import bbox_iou, bbox_iou_cat_in_lot
from utils.torch_utils import de_parallel
from utils.general import default_vlot_depth, default_hlot_depth, default_hlot_min_width, base_image_size, PI

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

with open(ROOT / '../data/hyps/hyp.scratch-low.yaml', errors='ignore') as f:
    hyp = yaml.safe_load(f)

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        print("!!!!!!!!!!!!!!!!!!!!!!!!",device)
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        MSEnone = nn.MSELoss(reduction='none')
        MSEmean = nn.MSELoss(reduction='mean')
        MSEthetaAD = nn.MSELoss(reduction='none') #alreadty test nn.L1Loss(reduction='none') is not as good as like nn.MSELoss(reduction='none')
        MSEobj = nn.MSELoss(reduction='none')

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets
        
        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        #self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.balance = {3: [1.0, 1.0, 1.0]}.get(m.nl, [1.0, 1.0, 1.0, 1.0, 1.0])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.MSEnone, self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = MSEnone, BCEcls, BCEobj, 1.0, h, autobalance
        self.MSEmean = MSEmean
        self.MSEobj = MSEobj
        self.MSEthetaAD = MSEthetaAD
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
        
        self.polycar = np.array([270/640.0, 193/640.0, 370/640.0, 193/640.0, 370/640.0, 445/640.0, 270/640.0, 445/640.0]).reshape(4, 2)  # 四边形二维坐标表示
        self.polycar = Polygon(self.polycar).convex_hull
        self.consum_time=[0,0,0,0]

    def __call__(self, p, targets, epoch=0, epochs=1200, i_iter=0):  # predictions, targets
        need_cal = epoch<=1
        if(i_iter==0 and need_cal):
            self.consum_time=[0,0,0,0]
        if (need_cal):
            function_start_t = perf_counter_ns()
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        #tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        if (need_cal):
            start = perf_counter_ns()
        tcls, Atbox, Aindices, Btbox, Bindices, anchors = self.build_targets(p, targets)
        if (need_cal):
            end = perf_counter_ns()
            self.consum_time[0] += end-start
        random.seed(time.time_ns()%(2**32 - 1))
        # random.seed(epoch + i_iter) #原本是想着DDP，确保所有进程选取相同位置的负样本，但是由于正样本没有统一，所以没什么用
        
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            rowlist = range(pi.shape[2]-1)
            collist = range(pi.shape[3]-1)
            Ab, Aa, Agj, Agi = Aindices[i]  # image, anchor, gridy, gridx
            Bb, Ba, Bgj, Bgi = Bindices[i]  # image, anchor, gridy, gridx

            Atobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj
            Btobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            #Atobj = torch.full_like(Atobj, self.cn, device=self.device)
            #Btobj = torch.full_like(Btobj, self.cn, device=self.device)
            
            na = Ab.shape[0]
            nb = Bb.shape[0]
            n = na + nb  # number of targets
            if (need_cal):
                start = perf_counter_ns()
            if n:
                #                                    0  1  2   3    4   5   6    7    8  9  10  11   12  13  14   15   16   17
                #target-subset of predictions #pred: Ax Ay Ac1 As1 Ac2 As2 Alen Aobj  Bx By Bc1 Bs1  Bc2 Bs2 Blen Bobj cls1 cls2
                Apxy, Apot, Aobj, _, _    = pi[Ab, Aa, Agj, Agi].split((2, 5, 1, 8, self.nc), 1)  
                _, Bpxy, Bpot, Bobj, pcls = pi[Bb, Ba, Bgj, Bgi].split((8, 2, 5, 1, self.nc), 1) 
                
                Aitst = Atbox[i][:,-1:]
                Atbox[i] = Atbox[i][:,0:-1]
                Bitst = Btbox[i][:,-1:]
                Btbox[i] = Btbox[i][:,0:-1]
                
                if (hyp["ALL_PARKING_LOT_SAME_WEIGHT"]):
                    #顺序不能倒过来，必须先设置true,再设置false
                    Aitst[Aitst==True] = 5
                    Aitst[Aitst==False] = 1
                    Bitst[Bitst==True] = 5
                    Bitst[Bitst==False] = 1
                else:
                    Aitst[Aitst>0.25] = 1
                    Aitst[Aitst>0] *= 5
                    Aitst += 1
                    Bitst[Bitst>0.25] = 1
                    Bitst[Bitst>0] *= 5
                    Bitst += 1
                
                # Aitst = torch.ones_like(Aitst, device=self.device)
                # Bitst = torch.ones_like(Bitst, device=self.device)
                Aitst=torch.concat((Aitst,Aitst),axis=1)
                Bitst=torch.concat((Bitst,Bitst),axis=1)
                
                
                # Regression
                if hyp["USE_THREE_POSITIVE_SAMPLE"]:
                    Apxy = 2 * Apxy.sigmoid() - 0.5
                    Bpxy = 2 * Bpxy.sigmoid() - 0.5
                else:
                    Apxy = Apxy.sigmoid()
                    Bpxy = Bpxy.sigmoid()
                
                Apot = torch.cat( (Apot[:,0:4].tanh() , Apot[:,4:5].sigmoid() ) , dim=1 )
                Bpot = torch.cat( (Bpot[:,0:4].tanh() , Bpot[:,4:5].sigmoid() ) , dim=1 )
                
                Apbox = torch.cat((Apxy, Apot), 1)
                Bpbox = torch.cat((Bpxy, Bpot), 1) 
                
                # hlot not care about direction theta, vlot&islot care about direction theta  0.375=4.8/12.8
                AweightstheAD = torch.zeros((na,1), dtype=pi.dtype, device=self.device)+1.5
                AweightstheAD[Atbox[i][:,6:7]>0.375] = 0.25
                AweightstheAD=torch.concat((AweightstheAD,AweightstheAD),axis=1)
                
                BweightstheAD = torch.zeros((nb,1), dtype=pi.dtype, device=self.device)+1.5
                BweightstheAD[Btbox[i][:,6:7]>0.375] = 0.25
                BweightstheAD=torch.concat((BweightstheAD,BweightstheAD),axis=1)
                
                #if(torch.sum(Atbox[i][:,6:7]>0.375)>0):
                #    print("na,nb,n: ",na,nb,n)
                #    print("AweightstheAD: ",AweightstheAD)
                #    print("BweightstheAD: ",BweightstheAD)
                #    sys.exit()
                
                
                lossxy  = torch.sum(self.MSEnone(Apbox[:,0:2], Atbox[i][:,0:2]) * Aitst) / (na*2) \
                        + torch.sum(self.MSEnone(Bpbox[:,0:2], Btbox[i][:,0:2]) * Bitst) / (nb*2)
                
                if 0: #cos 和 sin 是否先归一化
                    Apbox_normal = torch.nn.functional.normalize(Apbox[:,2:4], dim=1, eps=1e-12)
                    Bpbox_normal = torch.nn.functional.normalize(Bpbox[:,2:4], dim=1, eps=1e-12)
                else:
                    Apbox_normal = Apbox[:,2:4] 
                    Bpbox_normal = Bpbox[:,2:4] 

                if 0:  #cos 和 sin 是否带权重后计算L2loss 
                    Aweight = torch.pow((1.5 - torch.abs(Atbox[i][:,2:4])), 2)
                    Bweight = torch.pow((1.5 - torch.abs(Btbox[i][:,2:4])), 2)
                    losstheAD =  torch.sum(self.MSEthetaAD(Apbox_normal, Atbox[i][:,2:4]) * Aweight * Aitst * AweightstheAD) / (na*2) \
                              +  torch.sum(self.MSEthetaAD(Bpbox_normal, Btbox[i][:,2:4]) * Bweight * Bitst * BweightstheAD) / (nb*2)
                elif 0: 
                    if 0: #直接使用余弦相似度 有一个问题就是即使余弦相似度到了0.9999，弧度0.014142253477512098，弧度差距还是蛮大的，不符合库位检测精度要求
                        Acos_sim = torch.cosine_similarity(Apbox_normal, Atbox[i][:,2:4], eps=1e-6, dim=1)
                        Bcos_sim = torch.cosine_similarity(Bpbox_normal, Btbox[i][:,2:4], eps=1e-6, dim=1)
                        Acos_sim_gt = torch.ones(Apbox_normal.shape[0], device=self.device)
                        Bcos_sim_gt = torch.ones(Bpbox_normal.shape[0], device=self.device)
                        losstheAD =  torch.sum(torch.nn.functional.smooth_l1_loss(Acos_sim, Acos_sim_gt, reduction='none') * Aitst[:,0] * AweightstheAD[:,0]) / (na) \
                                +  torch.sum(torch.nn.functional.smooth_l1_loss(Bcos_sim, Bcos_sim_gt, reduction='none') * Bitst[:,0] * BweightstheAD[:,0]) / (nb)
                    else: #防止余弦相似度在0附近导数为0的情况，这里直接用角度去计算loss
                        inf_ = 1e-6
                        Acos_sim = torch.acos(torch.clip(torch.cosine_similarity(Apbox_normal, Atbox[i][:,2:4], eps=1e-6, dim=1), min=-1+inf_, max=1-inf_))
                        Bcos_sim = torch.acos(torch.clip(torch.cosine_similarity(Bpbox_normal, Btbox[i][:,2:4], eps=1e-6, dim=1), min=-1+inf_, max=1-inf_))
                        Acos_sim_gt = torch.zeros(Apbox_normal.shape[0], device=self.device)
                        Bcos_sim_gt = torch.zeros(Bpbox_normal.shape[0], device=self.device)
                        losstheAD =  torch.sum(torch.nn.functional.smooth_l1_loss(Acos_sim, Acos_sim_gt, reduction='none') * Aitst[:,0] * AweightstheAD[:,0]) / (na) \
                                +  torch.sum(torch.nn.functional.smooth_l1_loss(Bcos_sim, Bcos_sim_gt, reduction='none') * Bitst[:,0] * BweightstheAD[:,0]) / (nb)
                    losstheAD *= 2.4 #结合后面的*2，这里等同于乘以4.8，约等于endpose到入口线的距离，通过这种方式平衡欧式距离误差和角度误差的权重
                else: #cos 和 sin 直接计算L2loss
                    losstheAD =  torch.sum(self.MSEthetaAD(Apbox_normal, Atbox[i][:,2:4]) * Aitst * AweightstheAD) / (na*2) \
                              +  torch.sum(self.MSEthetaAD(Bpbox_normal, Btbox[i][:,2:4]) * Bitst * BweightstheAD) / (nb*2)
                                        
                losstheAB = torch.sum(self.MSEnone(Apbox[:,4:6], Atbox[i][:,4:6])) / (na*2) \
                          + torch.sum(self.MSEnone(Bpbox[:,4:6], Btbox[i][:,4:6])) / (nb*2)
                # losstheAD = torch.sum(self.MSEthetaAD(torch.atan2(Apbox[:,2:3],Apbox[:,3:4]) , torch.atan2(Atbox[i][:,2:3],Atbox[i][:,3:4])) * Aitst[:,0:1] * AweightstheAD) / na  \
                #           + torch.sum(self.MSEthetaAD(torch.atan2(Bpbox[:,2:3],Bpbox[:,3:4]) , torch.atan2(Btbox[i][:,2:3],Btbox[i][:,3:4])) * Bitst[:,0:1] * BweightstheAD) / nb 
                # losstheAB = torch.sum(self.MSEthetaAD(torch.atan2(Apbox[:,4:5],Apbox[:,5:6]) , torch.atan2(Atbox[i][:,4:5],Atbox[i][:,5:6]))) / na  \
                #           + torch.sum(self.MSEthetaAD(torch.atan2(Bpbox[:,4:5],Bpbox[:,5:6]) , torch.atan2(Btbox[i][:,4:5],Btbox[i][:,5:6]))) / nb 
                          
                losslen = torch.sum(self.MSEnone(Apbox[:,6:7], Atbox[i][:,6:7])) / (na) \
                          + torch.sum(+ self.MSEnone(Bpbox[:,6:7], Btbox[i][:,6:7])) / (nb)
                
                lbox += lossxy * 1 + losslen * 0.75 + losstheAB * 0.75 + losstheAD * 2
                    
                #iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                #lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                #iou = iou.detach().clamp(0).type(tobj.dtype)
                #if self.sort_obj_iou:
                #    j = iou.argsort()
                #    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                #if self.gr < 1:
                #    iou = (1.0 - self.gr) + self.gr * iou   
                Atobj[Ab, Aa, Agj, Agi] = 1  # iou ratio
                Btobj[Bb, Ba, Bgj, Bgi] = 1  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(nb), tcls[i]] = self.cp
                    lcls += self.MSEmean(pcls.sigmoid(), t) #self.BCEcls(pcls, t)  # BCE
                    
            if (need_cal):
                end = perf_counter_ns()
                self.consum_time[1] += end-start
                start = perf_counter_ns()
            
            if na:
                Aselect = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device) #NCHW
                if (epoch >= epochs-40):
                    _baseline_samp = 20 # hyp["NEG_POS_TOTAL"]
                    _bs = pi.shape[0]
                    _as = pi.shape[1]
                    _h  = pi.shape[2]
                    _w  = pi.shape[3]
                    
                    conftemp = pi[..., 7].sigmoid()
                    conftemp = conftemp.reshape(_bs, _as, -1)
                    idc = torch.sort(conftemp, -1, True).indices #降序排列, 挑选置信度最高的预测值
                    idc = idc[:, :, :_baseline_samp]
                    idcreshape = idc.reshape(-1)
                    
                    _bsrepeat = torch.arange(_bs).repeat_interleave(_as*_baseline_samp)
                    _asrepeat = torch.arange(_as).repeat(_bs).repeat_interleave(_baseline_samp)
                    hlist = idcreshape//_w
                    wlist = idcreshape%_w
                    Aselect[_bsrepeat, _asrepeat, hlist, wlist] = 1
                    Aselect[Ab, Aa, Agj, Agi] = 1 
                    # print("pi.shape: ", pi.shape)
                    # print("_bsrepeat: ", _bsrepeat)
                    # print("_asrepeat:", _asrepeat)
                    # print("hlist:", hlist)
                    # print("wlist:", wlist)
                    # print("Ab, Aa, Agj, Agi: ",Ab, Aa, Agj, Agi)
                    # print("torch.sum(Aselect): ", torch.sum(Aselect))
                    # df = pd.DataFrame(np.array(Aselect[0][0].cpu()))
                    # df.to_excel(r"11.xlsx", sheet_name="sheet1", index= False,encoding="utf-8")
                    # df = pd.DataFrame(np.array(Aselect[1][0].cpu()))
                    # df.to_excel(r"12.xlsx", sheet_name="sheet1", index= False,encoding="utf-8")
                    # sys.exit()
                else: #随机选择负样本
                    # 每个图像的每层archor(实际上就一个archor)上必须有_baseline_neg个负样本
                    # pi.shape[:4] batchsize anchor_num outputbuffer_h outputbuffer_w
                    _baseline_neg = 20
                    _bs = pi.shape[0]
                    _as = pi.shape[1]
                    batch_list = torch.arange(0, _bs).repeat(_baseline_neg*_as)
                    anchor_list = torch.arange(0, _as).repeat(_baseline_neg*_bs)
                    Aselect[batch_list, anchor_list, random.choices(rowlist, k = _baseline_neg*_as*_bs),random.choices(collist, k = _baseline_neg*_as*_bs)] = 1
                    list1 = random.choices(rowlist, k = na * hyp["NEG_POS_RATE"])
                    list2 = random.choices(collist, k = na * hyp["NEG_POS_RATE"])
                    Aselect[Ab.repeat(hyp["NEG_POS_RATE"]), Aa.repeat(hyp["NEG_POS_RATE"]), list1, list2] = 1
                    Aselect[Ab, Aa, Agj, Agi] = 1 
            else:
                Aselect = torch.ones(pi.shape[:4], dtype=pi.dtype, device=self.device)
            
            
            if nb:
                Bselect = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)

                if (epoch >= epochs-40):
                    _baseline_samp = 20 # hyp["NEG_POS_TOTAL"]
                    _bs = pi.shape[0]
                    _as = pi.shape[1]
                    _h  = pi.shape[2]
                    _w  = pi.shape[3]
                    
                    conftemp = pi[..., 15].sigmoid()
                    conftemp = conftemp.reshape(_bs, _as, -1)
                    idc = torch.sort(conftemp, -1, True).indices #降序排列, 挑选置信度最高的预测值
                    idc = idc[:, :, :_baseline_samp]
                    idcreshape = idc.reshape(-1)
                    
                    _bsrepeat = torch.arange(_bs).repeat_interleave(_as*_baseline_samp)
                    _asrepeat = torch.arange(_as).repeat(_bs).repeat_interleave(_baseline_samp)
                    hlist = idcreshape//_w
                    wlist = idcreshape%_w
                    Bselect[_bsrepeat, _asrepeat, hlist, wlist] = 1
                    Bselect[Bb, Ba, Bgj, Bgi] = 1
                else: #随机选择负样本
                    # 每个图像的每层archor(实际上就一个archor)上必须有_baseline_neg个负样本
                    _baseline_neg = 20
                    _bs = pi.shape[0]
                    _as = pi.shape[1]
                    batch_list = torch.arange(0, _bs).repeat(_baseline_neg*_as)
                    anchor_list = torch.arange(0, _as).repeat(_baseline_neg*_bs)
                    Bselect[batch_list, anchor_list, random.choices(rowlist, k = _baseline_neg*_as*_bs), random.choices(collist, k = _baseline_neg*_as*_bs)] = 1
                    
                    list1 = random.choices(rowlist, k = nb * hyp["NEG_POS_RATE"])
                    list2 = random.choices(collist, k = nb * hyp["NEG_POS_RATE"])
                    Bselect[Bb.repeat(hyp["NEG_POS_RATE"]), Ba.repeat(hyp["NEG_POS_RATE"]), list1, list2] = 1 
                    Bselect[Bb, Ba, Bgj, Bgi] = 1
                    
            else:
                Bselect = torch.ones(pi.shape[:4], dtype=pi.dtype, device=self.device)
            
            Aobji = torch.sum(self.MSEobj(pi[..., 7].sigmoid(), Atobj) * Aselect) / torch.sum(Aselect)
            Bobji = torch.sum(self.MSEobj(pi[..., 15].sigmoid(), Btobj) * Bselect) / torch.sum(Bselect)
            obji = Aobji + Bobji
            
            if (need_cal):
                end = perf_counter_ns()
                self.consum_time[2] += end-start
            
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = Atobj.shape[0]+Btobj.shape[0]  # batch size

        if (need_cal):
            function_end_t = perf_counter_ns()
            self.consum_time[3] += function_end_t - function_start_t
            
        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
    
    def get_cosume_time(self):
        print("cal loss use time disturb: ", [ele/1000000000.0 for ele in self.consum_time])
        
    def whether_lot_veh_intersects(self, targets):
        #                        A       B       C     D
        #           0      1   2   3   4   5   6   7  8  9
        #targets: img_id  cls  x1  y1  x2  y2  x3  y3 x4 y4
        temp_targets = copy.deepcopy(targets)
        if(temp_targets.shape[0]>0):
            lengGt = torch.full((temp_targets.shape[0] ,1), default_vlot_depth, device=self.device)
            widthGt = torch.sqrt((temp_targets[:,4] - temp_targets[:,2])**2+(temp_targets[:,5] - temp_targets[:,3])**2)
            hlotGT_idx = torch.tensor(range(temp_targets.shape[0]), device=self.device)[widthGt > (1.0*default_hlot_min_width)]
            lengGt[hlotGT_idx,0:1] = default_hlot_depth
            thetaBC = torch.atan2(temp_targets[:,7:8] - temp_targets[:,5:6],temp_targets[:,6:7] - temp_targets[:,4:5])
            thetaAD = torch.atan2(temp_targets[:,9:10] - temp_targets[:,3:4],temp_targets[:,8:9] - temp_targets[:,2:3])
            
            temp_targets[:,6:7] = torch.cos(thetaBC)*lengGt + temp_targets[:,4:5]
            temp_targets[:,7:8] = torch.sin(thetaBC)*lengGt + temp_targets[:,5:6]
            
            temp_targets[:,8:9] = torch.cos(thetaAD)*lengGt + temp_targets[:,2:3]
            temp_targets[:,9:10] = torch.sin(thetaAD)*lengGt + temp_targets[:,3:4]
            if (hyp["ALL_PARKING_LOT_SAME_WEIGHT"]):
                m = map(lambda lot : not self.polycar.disjoint(Polygon(lot[2:].reshape(4,2)).convex_hull), temp_targets.cpu() )
            else:
                m = map(lambda lot : bbox_iou_cat_in_lot(self.polycar, Polygon(lot[2:].reshape(4,2)).convex_hull ), temp_targets.cpu() )
            m = list(m)
        else:
            m=[]  
        return torch.tensor(m)  
        
    def build_targets(self, p, targets):   
        #translation
        #                        A       B       C     D
        #           0      1   2   3   4   5   6   7  8  9
        #targets: img_id  cls  x1  y1  x2  y2  x3  y3 x4 y4
        theta_AD = torch.atan2(targets[:,9:10] - targets[:,3:4],targets[:,8:9] - targets[:,2:3])
        theta_BC = torch.atan2(targets[:,7:8]  - targets[:,5:6],targets[:,6:7] - targets[:,4:5])
        theta_AB = torch.atan2(targets[:,5:6]  - targets[:,3:4],targets[:,4:5] - targets[:,2:3])
        theta_BA = torch.atan2(targets[:,3:4]  - targets[:,5:6],targets[:,2:3] - targets[:,4:5])

        lengt_AB = torch.sqrt(  torch.pow(targets[:,4:5] - targets[:,2:3],2) +  torch.pow(targets[:,5:6] - targets[:,3:4],2) )

        tmp = torch.cat((torch.zeros_like(targets,device=self.device),torch.zeros(targets.shape[0],7,device=self.device)),dim=1)
        tmp[:,0:4] = targets[:,0:4]
        tmp[:,4:5] = torch.cos(theta_AD)
        tmp[:,5:6] = torch.sin(theta_AD)
        tmp[:,6:7] = lengt_AB
        tmp[:,7:8] = torch.cos(theta_AB)
        tmp[:,8:9] = torch.sin(theta_AB)

        tmp[:,9:11]  = targets[:,4:6]
        tmp[:,11:12] = torch.cos(theta_BC)
        tmp[:,12:13] = torch.sin(theta_BC)
        tmp[:,13:14] = lengt_AB
        tmp[:,14:15] = torch.cos(theta_BA)
        tmp[:,15:16] = torch.sin(theta_BA)
        
        tmp[:,16] = self.whether_lot_veh_intersects(targets)
        targets=tmp
        #   0      1   2  3   4  5  6    7  8          9  10 11 12  13  14 15     16
        #img_id occupy Ax Ay c1 s1  leng c2 s2   &     Bx By c1 s1 leng c2 s2 intersects


        
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, Atbox, Btbox, Aindices, Bindices, anch = [], [], [], [], [], []

        # normalized to gridspace gain: img_id occupy Ax Ay c1 s1  leng c2 s2 && Bx By c1 s1 leng c2 s2 intersects arid
        gain = torch.ones(18, device=self.device)
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            
            gain[2:4] = torch.tensor(shape)[[3, 2]]  # xyxy gain  # x y len 
            gain[9:11] = torch.tensor(shape)[[3, 2]] 

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                # not use anchor to fliter
                #r = t[..., 4:6] / anchors[:, None]  # wh ratio
                #j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                #t = t[j]  # filter
                
                #r = t[...,6:7] / anchors[:, None]
                #j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']
                #t = t[j]  # filter
                
                #t = torch.reshape(t,(-1,t.shape[-1]))
                
                # Offsets
                r = t[...,4:5] / anchors[:, None]
                j = torch.max(r, 1 / r).max(2)[0] > -1000000000
                t = t[j]
                    
                if hyp["USE_THREE_POSITIVE_SAMPLE"]:#这个策略暂时还没有处理，考虑所泊库位的权重情况，所以目前不要使用这个策略
                    Agxy = t[:, 2:4]  # grid xy
                    Bgxy = t[:, 9:11]  # grid xy
                    Agxi = gain[[2, 3]] - Agxy  # inverse
                    Bgxi = gain[[9, 10]] - Bgxy  # inverse
                    
                    Aj, Ak = ((Agxy % 1 < g) & (Agxy > 1)).T
                    Al, Am = ((Agxi % 1 < g) & (Agxi > 1)).T
                    Aj = torch.stack((torch.ones_like(Aj), Aj, Ak, Al, Am))
                    
                    Bj, Bk = ((Bgxy % 1 < g) & (Bgxy > 1)).T
                    Bl, Bm = ((Bgxi % 1 < g) & (Bgxi > 1)).T
                    Bj = torch.stack((torch.ones_like(Bj), Bj, Bk, Bl, Bm))
                    
                    #idx = torch.concat((Aj,Bj))
                    At = t.repeat((5, 1, 1))[Aj]
                    Bt = t.repeat((5, 1, 1))[Bj]
                    Aoffsets = (torch.zeros_like(Agxy)[None] + off[:, None])[Aj]
                    Boffsets = (torch.zeros_like(Bgxy)[None] + off[:, None])[Bj]
                    #t = torch.concat((At,Bt))
                    #offsets = torch.concat((Aoffsets,Boffsets))
                    # print("At.shape:",At.shape)
                    # print("Bt.shape:",Bt.shape)
                    # print("Aoffsets.shape:",Aoffsets.shape)
                    # print("Boffsets.shape:",Boffsets.shape)
                    # print("t.shape:",t.shape)
                    # print("offsets.shape:",offsets.shape)
                    # sys.exit()
                else:
                    offsets = 0
            else:
                t = targets[0]
                offsets = 0

            # Define
            ##   0      1   2  3   4  5  6    7  8          9  10 11 12  13  14 15
            #img_id occupy Ax Ay c1 s1  leng c2 s2   &     Bx By c1 s1 leng c2 s2
            if hyp["USE_THREE_POSITIVE_SAMPLE"]:
                tmp_Ab, tmp_Ac, tmp_Ax,tmp_Ay,tmp_Ac1, tmp_As1, tmp_Aleng, tmp_Ac2, tmp_As2, _, _, _, _, _, _, _, Aa = At.chunk(17, 1)
                tmp_Bb, tmp_Bc, _, _, _, _, _, _, _, tmp_Bx, tmp_By, tmp_Bc1, tmp_Bs1, tmp_Bleng, tmp_Bc2, tmp_Bs2, Ba = Bt.chunk(17, 1)
                Abc = torch.concat((tmp_Ab,tmp_Ac), dim=1)
                Bbc = torch.concat((tmp_Bb,tmp_Bc), dim=1)
                Agxy = torch.concat((tmp_Ax,tmp_Ay), dim=1)
                Agot = torch.concat((tmp_Ac1,tmp_As1, tmp_Ac2, tmp_As2, tmp_Aleng), dim=1)
                Bgxy = torch.concat((tmp_Bx,tmp_By), dim=1)
                Bgot = torch.concat((tmp_Bc1,tmp_Bs1, tmp_Bc2, tmp_Bs2, tmp_Bleng), dim=1)
                
                Aa, (Ab, Ac) = Aa.long().view(-1), Abc.long().T  # anchors, imageID, class
                Ba, (Bb, Bc) = Ba.long().view(-1), Bbc.long().T  # anchors, imageID, class
                
                Agij = (Agxy - Aoffsets).long()
                Agi, Agj = Agij.T  # grid indices

                Bgij = (Bgxy - Boffsets).long()
                Bgi, Bgj = Bgij.T  # grid indices
                
                # Append
                Aindices.append((Ab, Aa, Agj.clamp_(0, shape[2] - 1), Agi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
                Bindices.append((Bb, Ba, Bgj.clamp_(0, shape[2] - 1), Bgi.clamp_(0, shape[3] - 1)))  # image, anchor, grid

                Atbox.append(torch.cat((Agxy - Agij, Agot), 1))  # box
                Btbox.append(torch.cat((Bgxy - Bgij, Bgot), 1))  # box

                anch.append(anchors[Aa])  # anchors
                tcls.append(Bc)  # class
                # print("Btbox: ", Btbox)
                # print("Bindices: ", Bindices)
                # print("Bc: ",Bc)
                # sys.exit()

            else:
                tmp_b, tmp_c, tmp_Ax,tmp_Ay,tmp_Ac1, tmp_As1, tmp_Aleng, tmp_Ac2, tmp_As2, tmp_Bx, tmp_By, tmp_Bc1, tmp_Bs1, tmp_Bleng, tmp_Bc2, tmp_Bs2, itst, a = t.chunk(18, 1)
                bc = torch.concat((tmp_b,tmp_c), dim=1)
                Agxy = torch.concat((tmp_Ax,tmp_Ay), dim=1)
                Agot = torch.concat((tmp_Ac1,tmp_As1, tmp_Ac2, tmp_As2, tmp_Aleng, itst), dim=1)
                Bgxy = torch.concat((tmp_Bx,tmp_By), dim=1)
                Bgot = torch.concat((tmp_Bc1,tmp_Bs1, tmp_Bc2, tmp_Bs2, tmp_Bleng, itst), dim=1)
                
                a, (b, c) = a.long().view(-1), bc.long().T  # anchors, imageID, class
                
                Agij = (Agxy - offsets).long()
                Agi, Agj = Agij.T  # grid indices

                Bgij = (Bgxy - offsets).long()
                Bgi, Bgj = Bgij.T  # grid indices

                # Append
                Aindices.append((b, a, Agj.clamp_(0, shape[2] - 1), Agi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
                Bindices.append((b, a, Bgj.clamp_(0, shape[2] - 1), Bgi.clamp_(0, shape[3] - 1)))  # image, anchor, grid

                Atbox.append(torch.cat((Agxy - Agij, Agot), 1))  # box
                Btbox.append(torch.cat((Bgxy - Bgij, Bgot), 1))  # box

                anch.append(anchors[a])  # anchors
                tcls.append(c)  # class 
        # time.sleep(10)
        # print("tcls: ",tcls)
        # print("Atbox: ",Atbox)
        # print("Aindices: ",Aindices)
        # print("Btbox: ",Btbox)
        # print("Bindices: ",Bindices)
        # print("anch: ",anch)
        # sys.exit()
        return tcls, Atbox, Aindices, Btbox, Bindices, anch
