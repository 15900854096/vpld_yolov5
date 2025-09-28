# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import sys
import time
import yaml
import os
import random
import copy
from pathlib import Path
import pandas as pd
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPoint
from time import perf_counter_ns
from utils.metrics import bbox_iou, bbox_iou_cat_in_lot
from utils.torch_utils import de_parallel
from utils.general import default_vlot_depth, default_hlot_depth, default_hlot_min_width, base_image_size, PI
from functools import reduce

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

priflag=True
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
        self.focalmean = MSEmean #FocalLoss(nn.BCEWithLogitsLoss(reduction='mean'), gamma=5, alpha=0.9)
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
        
        self.polycar = np.array([270/640.0, 193/640.0, 370/640.0, 193/640.0, 370/640.0, 445/640.0, 270/640.0, 445/640.0]).reshape(4, 2)  # 四边形二维坐标表示
        self.polycar = Polygon(self.polycar).convex_hull
        self.consum_time=[0,0,0,0]

    def __call__(self, p, targets, epoch=0, epochs=1200, i_iter=0, RANK=-1):  # predictions, targets
        need_cal = epoch<=1
        if(i_iter==0 and need_cal):
            self.consum_time=[0,0,0,0]
        if (need_cal):
            function_start_t = perf_counter_ns()
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        #tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        
        open_padding_str = False
        if (need_cal):
            start = perf_counter_ns()
        tcls, Atbox, Aindices, Btbox, Bindices, Ctbox, Cindices, Dtbox, Dindices, anchors, Ctbox_pdding, Cindices_padding, Dtbox_padding, Dindices_padding = self.build_targets(p, targets, open_padding_str)
        # print("Ctbox: ",Ctbox)
        # print("Cindices: ",Cindices)
        if (need_cal):
            end = perf_counter_ns()
            self.consum_time[0] += end-start

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions 这里只是对小中大三个图层进行剥离
            
            rowlist = range(pi.shape[2])
            collist = range(pi.shape[3])
            Ab, Aa, Agj, Agi = Aindices[i]  # image, anchor, gridy, gridx
            Bb, Ba, Bgj, Bgi = Bindices[i]  # image, anchor, gridy, gridx
            Cb, Ca, Cgj, Cgi = Cindices[i]  # image, anchor, gridy, gridx
            Db, Da, Dgj, Dgi = Dindices[i]  # image, anchor, gridy, gridx
            
            Atobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)+0.1  # target obj
            Btobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)+0.1  # target obj
            Ctobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)+0.1  # target obj
            Dtobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)+0.1  # target obj

            #Atobj = torch.full_like(Atobj, self.cn, device=self.device)
            #Btobj = torch.full_like(Btobj, self.cn, device=self.device)
            
            nA = Ab.shape[0]
            nB = Bb.shape[0]
            nC = Cb.shape[0]
            nD = Db.shape[0]
            assert( (nA==nB) and (nA==nC) and (nA==nD) )
            n = nA + nB + nC + nD  # number of targets
            if (need_cal):
                start = perf_counter_ns()
            if n:
                #                                    0  1  2   3    4   5   6    7    8  9  10  11   12  13  14   15   16   17   18 19 20 21  22   23   24 25 26 27  28   29   
                #target-subset of predictions #pred: Ax Ay Ac1 As1 Ac2 As2 Alen Aobj  Bx By Bc1 Bs1  Bc2 Bs2 Blen Bobj cls1 cls2 Cx Cy Cc Cs Clen Cobj  Dx Dy Dc Ds Dlen Dobj
                Apxy, Apot, Aobj, _, _, _ = pi[Ab, Aa, Agj, Agi].split((2, 5, 1, 8, self.nc, 12), 1)  
                _, Bpxy, Bpot, Bobj, pcls, _ = pi[Bb, Ba, Bgj, Bgi].split((8, 2, 5, 1, self.nc, 12), 1) 
                _, Cpxy, Cpot, Cobj, _ = pi[Cb, Ca, Cgj, Cgi].split((16 + self.nc, 2, 3, 1, 6), 1) 
                _, Dpxy, Dpot, Dobj = pi[Db, Da, Dgj, Dgi].split((16 + self.nc + 6, 2, 3, 1), 1) 
                    
                Aitst = Atbox[i][:,-1:]
                Atbox[i] = Atbox[i][:,0:-1]
                Bitst = Btbox[i][:,-1:]
                Btbox[i] = Btbox[i][:,0:-1]
                Citst = Ctbox[i][:,-1:]
                Ctbox[i] = Ctbox[i][:,0:-1]
                Ditst = Dtbox[i][:,-1:]
                Dtbox[i] = Dtbox[i][:,0:-1]
                
                if (hyp["ALL_PARKING_LOT_SAME_WEIGHT"]):
                    #顺序不能倒过来，必须先设置true,再设置false
                    Aitst[Aitst==True] = 5
                    Aitst[Aitst==False] = 1
                    Bitst[Bitst==True] = 5
                    Bitst[Bitst==False] = 1
                    Citst[Citst==True] = 5
                    Citst[Citst==False] = 1
                    Ditst[Ditst==True] = 5
                    Ditst[Ditst==False] = 1
                else:
                    Aitst[Aitst>0.25] = 1
                    Aitst[Aitst>0] *= 5
                    Aitst += 1
                    Bitst[Bitst>0.25] = 1
                    Bitst[Bitst>0] *= 5
                    Bitst += 1
                    Citst[Citst>0.25] = 1
                    Citst[Citst>0] *= 5
                    Citst += 1
                    Ditst[Ditst>0.25] = 1
                    Ditst[Ditst>0] *= 5
                    Ditst += 1
                
                Aitst=torch.concat((Aitst,Aitst),axis=1)
                Bitst=torch.concat((Bitst,Bitst),axis=1)
                Citst=torch.concat((Citst,Citst),axis=1)
                Ditst=torch.concat((Ditst,Ditst),axis=1)
                
                # Regression
                if hyp["USE_THREE_POSITIVE_SAMPLE"]:
                    Apxy = 2 * Apxy.sigmoid() - 0.5
                    Bpxy = 2 * Bpxy.sigmoid() - 0.5
                else:
                    Apxy = Apxy.sigmoid()
                    Bpxy = Bpxy.sigmoid()
                    Cpxy = Cpxy.sigmoid()
                    Dpxy = Dpxy.sigmoid()
                
                Apot = torch.cat( (Apot[:,0:4].tanh() , Apot[:,4:5].sigmoid() ) , dim=1 )
                Bpot = torch.cat( (Bpot[:,0:4].tanh() , Bpot[:,4:5].sigmoid() ) , dim=1 )
                Cpot = torch.cat( (Cpot[:,0:2].tanh() , Cpot[:,2:3].sigmoid() ) , dim=1 )
                Dpot = torch.cat( (Dpot[:,0:2].tanh() , Dpot[:,2:3].sigmoid() ) , dim=1 )
                    
                Apbox = torch.cat((Apxy, Apot), 1)
                Bpbox = torch.cat((Bpxy, Bpot), 1) 
                Cpbox = torch.cat((Cpxy, Cpot), 1)
                Dpbox = torch.cat((Dpxy, Dpot), 1) 
                
                # hlot not care about direction theta, vlot&islot care about direction theta  0.375=4.8/12.8
                AweightstheAD = torch.zeros((nA,1), dtype=pi.dtype, device=self.device)+1.5
                AweightstheAD[Atbox[i][:,6:7]>0.375] = 0.25
                AweightstheAD=torch.concat((AweightstheAD,AweightstheAD),axis=1)
                
                BweightstheAD = torch.zeros((nB,1), dtype=pi.dtype, device=self.device)+1.5
                BweightstheAD[Btbox[i][:,6:7]>0.375] = 0.25
                BweightstheAD=torch.concat((BweightstheAD,BweightstheAD),axis=1)
                
                #所泊库位与其他库位重要性不一样，需要有所平衡，通过Aitst调节
                #CD角点的位置由于水平和垂直的缘故，重要性也不一样，通过BweightstheAD来调节
                #AB角点位置不论水平还是垂直，都很重要
                lossxyA = torch.sum(self.MSEnone(Apbox[:,0:2], Atbox[i][:,0:2]) * Aitst) / (nA*2)
                lossxyB = torch.sum(self.MSEnone(Bpbox[:,0:2], Btbox[i][:,0:2]) * Bitst) / (nB*2)
                lossxyC = torch.sum(self.MSEnone(Cpbox[:,0:2], Ctbox[i][:,0:2]) * Citst * BweightstheAD) / (nC*2)
                lossxyD = torch.sum(self.MSEnone(Dpbox[:,0:2], Dtbox[i][:,0:2]) * Ditst * AweightstheAD) / (nD*2)
                lossxyABCD  = lossxyA + lossxyB + lossxyC + lossxyD
                            

                # if 0: #cos 和 sin 是否先归一化
                #     Apbox_normal = torch.nn.functional.normalize(Apbox[:,2:4], dim=1, eps=1e-12)
                #     Bpbox_normal = torch.nn.functional.normalize(Bpbox[:,2:4], dim=1, eps=1e-12)
                # else:
                #     Apbox_normal = Apbox[:,2:4] 
                #     Bpbox_normal = Bpbox[:,2:4] 

                # if 0:  #cos 和 sin 是否带权重后计算L2loss 
                #     Aweight = torch.pow((1.5 - torch.abs(Atbox[i][:,2:4])), 2)
                #     Bweight = torch.pow((1.5 - torch.abs(Btbox[i][:,2:4])), 2)
                #     losstheAD =  torch.sum(self.MSEnone(Apbox_normal, Atbox[i][:,2:4]) * Aweight * Aitst * AweightstheAD) / (nA*2) \
                #               +  torch.sum(self.MSEnone(Bpbox_normal, Btbox[i][:,2:4]) * Bweight * Bitst * BweightstheAD) / (nB*2)
                # elif 1: 
                #     if 0: #直接使用余弦相似度 有一个问题就是即使余弦相似度到了0.9999，弧度0.014142253477512098，弧度差距还是蛮大的，不符合库位检测精度要求
                #         Acos_sim = torch.cosine_similarity(Apbox_normal, Atbox[i][:,2:4], eps=1e-6, dim=1)
                #         Bcos_sim = torch.cosine_similarity(Bpbox_normal, Btbox[i][:,2:4], eps=1e-6, dim=1)
                #         Acos_sim_gt = torch.ones(Apbox_normal.shape[0], device=self.device)
                #         Bcos_sim_gt = torch.ones(Bpbox_normal.shape[0], device=self.device)
                #         losstheAD =  torch.sum(torch.nn.functional.smooth_l1_loss(Acos_sim, Acos_sim_gt, reduction='none') * Aitst[:,0] * AweightstheAD[:,0]) / (nA) \
                #                 +  torch.sum(torch.nn.functional.smooth_l1_loss(Bcos_sim, Bcos_sim_gt, reduction='none') * Bitst[:,0] * BweightstheAD[:,0]) / (nB)
                #     else: #防止余弦相似度在0附近导数为0的情况，这里直接用角度去计算loss
                #         inf_ = 1e-6
                #         Acos_sim = torch.acos(torch.clip(torch.cosine_similarity(Apbox_normal, Atbox[i][:,2:4], eps=1e-6, dim=1), min=-1+inf_, max=1-inf_))
                #         Bcos_sim = torch.acos(torch.clip(torch.cosine_similarity(Bpbox_normal, Btbox[i][:,2:4], eps=1e-6, dim=1), min=-1+inf_, max=1-inf_))
                #         Acos_sim_gt = torch.zeros(Apbox_normal.shape[0], device=self.device)
                #         Bcos_sim_gt = torch.zeros(Bpbox_normal.shape[0], device=self.device)
                #         losstheAD =  torch.sum(torch.nn.functional.smooth_l1_loss(Acos_sim, Acos_sim_gt, reduction='none') * Aitst[:,0] * AweightstheAD[:,0]) / (nA) \
                #                 +  torch.sum(torch.nn.functional.smooth_l1_loss(Bcos_sim, Bcos_sim_gt, reduction='none') * Bitst[:,0] * BweightstheAD[:,0]) / (nB)
                #     losstheAD *= 2.4 #结合后面的*2，这里等同于乘以4.8，约等于endpose到入口线的距离，通过这种方式平衡欧式距离误差和角度误差的权重
                # else: #cos 和 sin 直接计算L2loss
                #     losstheAD =  torch.sum(self.MSEnone(Apbox_normal, Atbox[i][:,2:4]) * Aitst * AweightstheAD) / (nA*2) \
                #               +  torch.sum(self.MSEnone(Bpbox_normal, Btbox[i][:,2:4]) * Bitst * BweightstheAD) / (nB*2)

                #A->D B->C C->B D->A角度对于所泊库位|其他库位、垂直库位|水平库位都有一定关系，侧重点要有所区别
                losstheAD = torch.sum(self.MSEnone(Apbox[:,2:4], Atbox[i][:,2:4]) * Aitst * AweightstheAD) / (nA*2) 
                losstheBC = torch.sum(self.MSEnone(Bpbox[:,2:4], Btbox[i][:,2:4]) * Bitst * BweightstheAD) / (nB*2)
                    
                losstheCB = torch.sum(self.MSEnone(Cpbox[:,2:4], Ctbox[i][:,2:4]) * Citst * BweightstheAD) / (nC*2)
                losstheDA = torch.sum(self.MSEnone(Dpbox[:,2:4], Dtbox[i][:,2:4]) * Ditst * AweightstheAD) / (nD*2)
                                            
                losstheAB = torch.sum(self.MSEnone(Apbox[:,4:6], Atbox[i][:,4:6]) * Aitst ) / (nA*2) 
                losstheBA = torch.sum(self.MSEnone(Bpbox[:,4:6], Btbox[i][:,4:6]) * Bitst ) / (nB*2)
                
                # losstheAD = torch.sum(self.MSEnone(torch.atan2(Apbox[:,2:3],Apbox[:,3:4]) , torch.atan2(Atbox[i][:,2:3],Atbox[i][:,3:4])) * Aitst[:,0:1] * AweightstheAD) / nA  \
                #           + torch.sum(self.MSEnone(torch.atan2(Bpbox[:,2:3],Bpbox[:,3:4]) , torch.atan2(Btbox[i][:,2:3],Btbox[i][:,3:4])) * Bitst[:,0:1] * BweightstheAD) / nB 
                # losstheAB = torch.sum(self.MSEnone(torch.atan2(Apbox[:,4:5],Apbox[:,5:6]) , torch.atan2(Atbox[i][:,4:5],Atbox[i][:,5:6]))) / nA  \
                #           + torch.sum(self.MSEnone(torch.atan2(Bpbox[:,4:5],Bpbox[:,5:6]) , torch.atan2(Btbox[i][:,4:5],Btbox[i][:,5:6]))) / nB 
                          
                losslenAB = torch.sum(self.MSEnone(Apbox[:,6:7], Atbox[i][:,6:7]) * Aitst ) / (nA) 
                losslenBA = torch.sum(self.MSEnone(Bpbox[:,6:7], Btbox[i][:,6:7]) * Bitst ) / (nB)
                losslenCB = torch.sum(self.MSEnone(Cpbox[:,4:5], Ctbox[i][:,4:5]) * Citst * BweightstheAD) / (nC)
                losslenDA = torch.sum(self.MSEnone(Dpbox[:,4:5], Dtbox[i][:,4:5]) * Ditst * AweightstheAD) / (nD)
                
                lbox += lossxyABCD * 1.5 + (losslenAB +losslenBA) * 0.75 + (losstheAB + losstheBA) * 0.75 + (losstheAD + losstheBC + losstheCB + losstheDA) * 0.75 + (losslenCB + losslenDA)*0.75
                    
                #iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                #lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                #iou = iou.detach().clamp(0).type(tobj.dtype)
                #if self.sort_obj_iou:
                #    j = iou.argsort()
                #    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                #if self.gr < 1:
                #    iou = (1.0 - self.gr) + self.gr * iou
                Atobj[Ab, Aa, Agj, Agi] = 0.9  # iou ratio
                Btobj[Bb, Ba, Bgj, Bgi] = 0.9  # iou ratio
                Ctobj[Cb, Ca, Cgj, Cgi] = 0.9  # iou ratio
                Dtobj[Db, Da, Dgj, Dgi] = 0.9  # iou ratio
                
                # Classification
                if 0: #self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(nB), tcls[i]] = self.cp
                    lcls += self.MSEmean(pcls.sigmoid(), t) #self.BCEcls(pcls, t)  # BCE
                    
            if (need_cal):
                end = perf_counter_ns()
                self.consum_time[1] += end-start
                start = perf_counter_ns()
            
        
            zer = torch.zeros(1, device=self.device)
            one = torch.zeros(1, device=self.device)+0.9
            Cobji_pad, Cxy_pad, Ccossin_pad, Clen_pad = zer, zer, zer, zer
            Dobji_pad, Dxy_pad, Dcossin_pad, Dlen_pad = zer, zer, zer, zer
            if (open_padding_str):
                
                cotherconf = pi[..., 23].sigmoid()
                dotherconf = pi[..., 29].sigmoid()
                
                Cb_pad, Ca_pad, Cgj_pad, Cgi_pad = Cindices_padding[i]  # image, anchor, gridy, gridx 
                Db_pad, Da_pad, Dgj_pad, Dgi_pad = Dindices_padding[i]  # image, anchor, gridy, gridx
                
                
                uni_Cb_pad, uni_Ca_pad, uni_Cgj_pad, uni_Cgi_pad = self.remove_duplicates(Cb_pad, Ca_pad, Cgj_pad, Cgi_pad)
                uni_Db_pad, uni_Da_pad, uni_Dgj_pad, uni_Dgi_pad = self.remove_duplicates(Db_pad, Da_pad, Dgj_pad, Dgi_pad)
                
                
                c_conf_pos = cotherconf[uni_Cb_pad, uni_Ca_pad, uni_Cgj_pad, uni_Cgi_pad]>0.6 #bool
                d_conf_pos = dotherconf[uni_Db_pad, uni_Da_pad, uni_Dgj_pad, uni_Dgi_pad]>0.6 #bool
                
                
                pos_uni_Cb_pad, pos_uni_Ca_pad, pos_uni_Cgj_pad, pos_uni_Cgi_pad = uni_Cb_pad[c_conf_pos], uni_Ca_pad[c_conf_pos], uni_Cgj_pad[c_conf_pos], uni_Cgi_pad[c_conf_pos]
                pos_uni_Db_pad, pos_uni_Da_pad, pos_uni_Dgj_pad, pos_uni_Dgi_pad = uni_Db_pad[d_conf_pos], uni_Da_pad[d_conf_pos], uni_Dgj_pad[d_conf_pos], uni_Dgi_pad[d_conf_pos]
                
                tmp1 = torch.tensor([],device=self.device)
                tmp2 = torch.tensor([],device=self.device)
                tmp3 = torch.tensor([],device=self.device)
                for k in range(pos_uni_Cb_pad.shape[0]):
                    one_pos_uni_Cb_pad, one_pos_uni_Ca_pad, one_pos_uni_Cgj_pad, one_pos_uni_Cgi_pad = pos_uni_Cb_pad[k], pos_uni_Ca_pad[k], pos_uni_Cgj_pad[k], pos_uni_Cgi_pad[k]
                    boolselect = torch.logical_and(torch.logical_and(Cb_pad==one_pos_uni_Cb_pad , Ca_pad==one_pos_uni_Ca_pad),
                                                   torch.logical_and( Cgj_pad==one_pos_uni_Cgj_pad , Cgi_pad==one_pos_uni_Cgi_pad))

                    pre_pos_mutil_gt = Ctbox_pdding[i][boolselect] #x,y cos,sin,len
                    
                    dis = torch.sqrt(torch.pow(pi[one_pos_uni_Cb_pad, one_pos_uni_Ca_pad, one_pos_uni_Cgj_pad, one_pos_uni_Cgi_pad,18].sigmoid() -  pre_pos_mutil_gt[:,0],2)
                                   + torch.pow(pi[one_pos_uni_Cb_pad, one_pos_uni_Ca_pad, one_pos_uni_Cgj_pad, one_pos_uni_Cgi_pad,19].sigmoid() -  pre_pos_mutil_gt[:,1],2))
                    _,indices = torch.min(dis, dim=0)
                    
                    #Cobji_pad += self.MSEmean(cotherconf[one_pos_uni_Cb_pad, one_pos_uni_Ca_pad, one_pos_uni_Cgj_pad, one_pos_uni_Cgi_pad],  one)
                    # Cxy_pad += self.MSEmean(pi[one_pos_uni_Cb_pad, one_pos_uni_Ca_pad, one_pos_uni_Cgj_pad, one_pos_uni_Cgi_pad,18:20].sigmoid(),  pre_pos_mutil_gt[indices,0:2])/pos_uni_Cb_pad.shape[0]*2
                    # Ccossin_pad += self.MSEmean(pi[one_pos_uni_Cb_pad, one_pos_uni_Ca_pad, one_pos_uni_Cgj_pad, one_pos_uni_Cgi_pad,20:22].sigmoid(),  pre_pos_mutil_gt[indices,2:4])/pos_uni_Cb_pad.shape[0]*2
                    # Clen_pad += self.MSEmean(pi[one_pos_uni_Cb_pad, one_pos_uni_Ca_pad, one_pos_uni_Cgj_pad, one_pos_uni_Cgi_pad,22].sigmoid(),  pre_pos_mutil_gt[indices,4])/pos_uni_Cb_pad.shape[0]
                    tmp1 = torch.cat((tmp1,pre_pos_mutil_gt[indices,0:2].unsqueeze(0)),0)
                    tmp2 = torch.cat((tmp2,pre_pos_mutil_gt[indices,2:4].unsqueeze(0)),0)
                    tmp3 = torch.cat((tmp3,pre_pos_mutil_gt[indices,4:5].unsqueeze(0)),0)
                
                Cxy_pad = self.MSEmean(pi[pos_uni_Cb_pad, pos_uni_Ca_pad, pos_uni_Cgj_pad, pos_uni_Cgi_pad,18:20].sigmoid(),  tmp1   )
                Ccossin_pad = self.MSEmean(pi[pos_uni_Cb_pad, pos_uni_Ca_pad, pos_uni_Cgj_pad, pos_uni_Cgi_pad,20:22].sigmoid(),  tmp2   )
                Clen_pad = self.MSEmean(pi[pos_uni_Cb_pad, pos_uni_Ca_pad, pos_uni_Cgj_pad, pos_uni_Cgi_pad,22:23].sigmoid(),  tmp3   )
                     
                
                tmp1 = torch.tensor([],device=self.device)
                tmp2 = torch.tensor([],device=self.device)
                tmp3 = torch.tensor([],device=self.device)  
                for k in range(pos_uni_Db_pad.shape[0]):
                    one_pos_uni_Db_pad, one_pos_uni_Da_pad, one_pos_uni_Dgj_pad, one_pos_uni_Dgi_pad = pos_uni_Db_pad[k], pos_uni_Da_pad[k], pos_uni_Dgj_pad[k], pos_uni_Dgi_pad[k]
                    boolselect = torch.logical_and(torch.logical_and(Db_pad==one_pos_uni_Db_pad , Da_pad==one_pos_uni_Da_pad),
                                                   torch.logical_and(Dgj_pad==one_pos_uni_Dgj_pad , Dgi_pad==one_pos_uni_Dgi_pad))
                    pre_pos_mutil_gt = Dtbox_padding[i][boolselect] #x,y cos,sin,len
                    
                    dis = torch.sqrt(torch.pow(pi[one_pos_uni_Db_pad, one_pos_uni_Da_pad, one_pos_uni_Dgj_pad, one_pos_uni_Dgi_pad,24].sigmoid() -  pre_pos_mutil_gt[:,0],2)
                                   + torch.pow(pi[one_pos_uni_Db_pad, one_pos_uni_Da_pad, one_pos_uni_Dgj_pad, one_pos_uni_Dgi_pad,25].sigmoid() -  pre_pos_mutil_gt[:,1],2))
                    _,indices = torch.min(dis, dim=0)
                    
                    #Dobji_pad += self.MSEmean(cotherconf[one_pos_uni_Db_pad, one_pos_uni_Da_pad, one_pos_uni_Dgj_pad, one_pos_uni_Dgi_pad],  one)
                    #Dxy_pad += self.MSEmean(pi[one_pos_uni_Db_pad, one_pos_uni_Da_pad, one_pos_uni_Dgj_pad, one_pos_uni_Dgi_pad,24:26].sigmoid(),  pre_pos_mutil_gt[indices,0:2])
                    #Dcossin_pad += self.MSEmean(pi[one_pos_uni_Db_pad, one_pos_uni_Da_pad, one_pos_uni_Dgj_pad, one_pos_uni_Dgi_pad,26:28].sigmoid(),  pre_pos_mutil_gt[indices,2:4])
                    #Dlen_pad += self.MSEmean(pi[one_pos_uni_Db_pad, one_pos_uni_Da_pad, one_pos_uni_Dgj_pad, one_pos_uni_Dgi_pad,28].sigmoid(),  pre_pos_mutil_gt[indices,4])
                    tmp1 = torch.cat((tmp1,pre_pos_mutil_gt[indices,0:2].unsqueeze(0)),0)
                    tmp2 = torch.cat((tmp2,pre_pos_mutil_gt[indices,2:4].unsqueeze(0)),0)
                    tmp3 = torch.cat((tmp3,pre_pos_mutil_gt[indices,4:5].unsqueeze(0)),0)
                Dxy_pad = self.MSEmean(pi[pos_uni_Db_pad, pos_uni_Da_pad, pos_uni_Dgj_pad, pos_uni_Dgi_pad,24:26].sigmoid(),  tmp1   )
                Dcossin_pad = self.MSEmean(pi[pos_uni_Db_pad, pos_uni_Da_pad, pos_uni_Dgj_pad, pos_uni_Dgi_pad,26:28].sigmoid(),  tmp2   )
                Dlen_pad = self.MSEmean(pi[pos_uni_Db_pad, pos_uni_Da_pad, pos_uni_Dgj_pad, pos_uni_Dgi_pad,28:29].sigmoid(),  tmp3   )
                   
                lbox += (Cxy_pad + Dxy_pad) * 1.5 + (Clen_pad + Dlen_pad) * 0.75 + (Ccossin_pad + Dcossin_pad) * 0.75
            
            if(hyp["MergeDiffSizeBufferDownsamplingFactor"] == 16):#OHEM or random select neg samples , for big buffer must use this select neg samples
                #每个图像的每层archor(实际上就一个archor)上必须有_baseline_neg个负样本
                # pi.shape[:4] batchsize anchor_num outputbuffer_h outputbuffer_w
                # random.seed(time.time_ns()%(2**32 - 1))
                # random.seed(epoch + i_iter)
                    
                _bs = pi.shape[0]
                _as = pi.shape[1]
                _h  = pi.shape[2]
                _w  = pi.shape[3]
                
                Aselect = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device) #NCHW
                Bselect = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)
                Cselect = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)
                Dselect = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)
                
                if(epoch<=epochs*0.7):#RANDOM
                    if(RANK!=-1): 
                        _baseline_neg = 20
                    else:
                        _baseline_neg = 20
                    batch_list = torch.arange(0, _bs).repeat(_baseline_neg*_as)
                    anchor_list = torch.arange(0, _as).repeat(_baseline_neg*_bs)
                    Aselect[batch_list, anchor_list, random.choices(rowlist, k = _baseline_neg*_as*_bs), random.choices(collist, k = _baseline_neg*_as*_bs)] = 1
                    if nA:
                        list1 = random.choices(rowlist, k = nA * hyp["NEG_POS_RATE"])
                        list2 = random.choices(collist, k = nA * hyp["NEG_POS_RATE"])
                        Aselect[Ab.repeat(hyp["NEG_POS_RATE"]), Aa.repeat(hyp["NEG_POS_RATE"]), list1, list2] = 1
                        Aselect[Ab, Aa, Agj, Agi] = 1 #hyp["NEG_POS_RATE"] 

                    batch_list = torch.arange(0, _bs).repeat(_baseline_neg*_as)
                    anchor_list = torch.arange(0, _as).repeat(_baseline_neg*_bs)
                    Bselect[batch_list, anchor_list, random.choices(rowlist, k = _baseline_neg*_as*_bs), random.choices(collist, k = _baseline_neg*_as*_bs)] = 1
                    if nB:
                        list1 = random.choices(rowlist, k = nB * hyp["NEG_POS_RATE"])
                        list2 = random.choices(collist, k = nB * hyp["NEG_POS_RATE"])
                        Bselect[Bb.repeat(hyp["NEG_POS_RATE"]), Ba.repeat(hyp["NEG_POS_RATE"]), list1, list2] = 1 
                        Bselect[Bb, Ba, Bgj, Bgi] = 1 #hyp["NEG_POS_RATE"]    
                    
                    batch_list = torch.arange(0, _bs).repeat(_baseline_neg*_as)
                    anchor_list = torch.arange(0, _as).repeat(_baseline_neg*_bs)
                    Cselect[batch_list, anchor_list, random.choices(rowlist, k = _baseline_neg*_as*_bs), random.choices(collist, k = _baseline_neg*_as*_bs)] = 1
                    if nC:
                        list1 = random.choices(rowlist, k = nC * hyp["NEG_POS_RATE"])
                        list2 = random.choices(collist, k = nC * hyp["NEG_POS_RATE"])
                        Cselect[Cb.repeat(hyp["NEG_POS_RATE"]), Ca.repeat(hyp["NEG_POS_RATE"]), list1, list2] = 1     
                        Cselect[Cb, Ca, Cgj, Cgi] = 1 #hyp["NEG_POS_RATE"] 
                
                    batch_list = torch.arange(0, _bs).repeat(_baseline_neg*_as)
                    anchor_list = torch.arange(0, _as).repeat(_baseline_neg*_bs)
                    Dselect[batch_list, anchor_list, random.choices(rowlist, k = _baseline_neg*_as*_bs), random.choices(collist, k = _baseline_neg*_as*_bs)] = 1
                    if nD:
                        list1 = random.choices(rowlist, k = nD * hyp["NEG_POS_RATE"])
                        list2 = random.choices(collist, k = nD * hyp["NEG_POS_RATE"])
                        Dselect[Db.repeat(hyp["NEG_POS_RATE"]), Da.repeat(hyp["NEG_POS_RATE"]), list1, list2] = 1
                        Dselect[Db, Da, Dgj, Dgi] = 1 #hyp["NEG_POS_RATE"]            
            
                else:#OHEM
                    if(RANK!=-1): 
                        _baseline_neg = 20
                    else:
                        _baseline_neg = 20
                    _bsrepeat = torch.arange(_bs).repeat_interleave(_as*_baseline_neg)
                    _asrepeat = torch.arange(_as).repeat(_bs).repeat_interleave(_baseline_neg)
                    
                    conftemp = pi[..., 7].sigmoid()
                    conftemp = conftemp.reshape(_bs, _as, -1)
                    idc = torch.sort(conftemp, -1, True).indices #降序排列, 挑选置信度最高的预测值
                    idc = idc[:, :, :_baseline_neg]
                    idcreshape = idc.reshape(-1)
                    hlist = idcreshape//_w
                    wlist = idcreshape%_w
                    Aselect[_bsrepeat, _asrepeat, hlist, wlist] = 1
                    Aselect[Ab, Aa, Agj, Agi] = 1    
                    
                    conftemp = pi[..., 15].sigmoid()
                    conftemp = conftemp.reshape(_bs, _as, -1)
                    idc = torch.sort(conftemp, -1, True).indices #降序排列, 挑选置信度最高的预测值
                    idc = idc[:, :, :_baseline_neg]
                    idcreshape = idc.reshape(-1)
                    hlist = idcreshape//_w
                    wlist = idcreshape%_w
                    Bselect[_bsrepeat, _asrepeat, hlist, wlist] = 1
                    Bselect[Bb, Ba, Bgj, Bgi] = 1 
                    
                    conftemp = pi[..., 23].sigmoid()
                    conftemp = conftemp.reshape(_bs, _as, -1)
                    idc = torch.sort(conftemp, -1, True).indices #降序排列, 挑选置信度最高的预测值
                    idc = idc[:, :, :_baseline_neg]
                    idcreshape = idc.reshape(-1)
                    hlist = idcreshape//_w
                    wlist = idcreshape%_w
                    Cselect[_bsrepeat, _asrepeat, hlist, wlist] = 1
                    Cselect[Cb, Ca, Cgj, Cgi] = 1 
                    
                    conftemp = pi[..., 29].sigmoid()
                    conftemp = conftemp.reshape(_bs, _as, -1)
                    idc = torch.sort(conftemp, -1, True).indices #降序排列, 挑选置信度最高的预测值
                    idc = idc[:, :, :_baseline_neg]
                    idcreshape = idc.reshape(-1)
                    hlist = idcreshape//_w
                    wlist = idcreshape%_w
                    Dselect[_bsrepeat, _asrepeat, hlist, wlist] = 1
                    Dselect[Db, Da, Dgj, Dgi] = 1 
                    
                if 0: #(RANK!=-1):    
                    gathered_tensors = [torch.empty_like(Aselect, device=f"cuda:{RANK}") for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_tensors, Aselect)
                    Aselect = reduce(lambda a,b:torch.logical_or(a, b),gathered_tensors)
                    
                    gathered_tensors = [torch.empty_like(Bselect, device=f"cuda:{RANK}") for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_tensors, Bselect)
                    Bselect = reduce(lambda a,b:torch.logical_or(a, b),gathered_tensors)
                    
                    gathered_tensors = [torch.empty_like(Cselect, device=f"cuda:{RANK}") for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_tensors, Cselect)
                    Cselect = reduce(lambda a,b:torch.logical_or(a, b),gathered_tensors)
                    
                    gathered_tensors = [torch.empty_like(Dselect, device=f"cuda:{RANK}") for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_tensors, Dselect)
                    Dselect = reduce(lambda a,b:torch.logical_or(a, b),gathered_tensors)
                    
                    # if(epoch==30 and i_iter==10):                        
                    #     df = pd.DataFrame(np.array(Aselect[0][0].cpu()))
                    #     df.to_excel(r"%d.xlsx"%RANK, sheet_name="sheet1", index= False,encoding="utf-8")
                    #     sys.exit()
                            
                Aobji = torch.sum(self.MSEnone(pi[..., 7].sigmoid(),  Atobj) * Aselect) / torch.sum(Aselect)
                Bobji = torch.sum(self.MSEnone(pi[..., 15].sigmoid(), Btobj) * Bselect) / torch.sum(Bselect)
                Cobji = torch.sum(self.MSEnone(pi[..., 23].sigmoid(), Ctobj) * Cselect) / torch.sum(Cselect) + Cobji_pad
                Dobji = torch.sum(self.MSEnone(pi[..., 29].sigmoid(), Dtobj) * Dselect) / torch.sum(Dselect) + Dobji_pad
            else:
                Aobji = self.focalmean(pi[..., 7].sigmoid(),   Atobj)*10
                Bobji = self.focalmean(pi[..., 15].sigmoid(),  Btobj)*10
                Cobji = self.focalmean(pi[..., 23].sigmoid(),  Ctobj)*10 + Cobji_pad
                Dobji = self.focalmean(pi[..., 29].sigmoid(),  Dtobj)*10 + Dobji_pad
            
            #if(epoch%5==4 and i_iter%50==0 and RANK==0):
            #    print(torch.sum(torch.round(pi[..., 7].sigmoid())).cpu(), "  ", torch.sum(torch.round(pi[..., 15].sigmoid())).cpu(), "  ", torch.sum(torch.round(pi[..., 23].sigmoid())).cpu(), "  ",torch.sum(torch.round(pi[..., 29].sigmoid())).cpu() )
            
            # np.set_printoptions(threshold=np.inf)
            # pd.set_option('display.width', 300) # 设置字符显示宽度
            # pd.set_option('display.max_rows', None) # 设置显示最大行
            # pd.set_option('display.max_columns', None) # 设置显示最大列，None为显示所有列
            # long_series = pd.Series(np.array(Dtobj[0,0,39,:].cpu()))
            # print(long_series)
            # df = pd.DataFrame(np.array(Ctobj[0][0].cpu()))
            # df.to_excel(r"11.xlsx", sheet_name="sheet1", index= False,encoding="utf-8")
            # df = pd.DataFrame(np.array(Cselect[0][0].cpu()))
            # df.to_excel(r"12.xlsx", sheet_name="sheet1", index= False,encoding="utf-8")
            # sys.exit()

            # print(Ctobj)
            # print(torch.sum(Ctobj))
            # print(torch.sum(Cselect))
            # print(Cb, Ca, Cgj, Cgi)
            # sys.exit()
            obji = Aobji + Bobji + Cobji + Dobji
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
        bs = Atobj.shape[0] + Btobj.shape[0] + Ctobj.shape[0] + Dtobj.shape[0]  # batch size

        if (need_cal):
            function_end_t = perf_counter_ns()
            self.consum_time[3] += function_end_t - function_start_t
            
        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
    
    def remove_duplicates(self, b_pad, a_pad, gj_pad, gi_pad):
        tmp = torch.stack((b_pad, a_pad, gj_pad, gi_pad), dim=1) 
        unique_b_pad_a_pad_gj_pad_gi_pad = torch.unique(tmp, dim=0)
        uni_b_pad, uni_a_pad, uni_gj_pad, uni_gi_pad = unique_b_pad_a_pad_gj_pad_gi_pad.T
        return uni_b_pad, uni_a_pad, uni_gj_pad, uni_gi_pad
    
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
    
    def connect(self, ends):
        d0, d1 = np.abs(np.diff(ends, axis=0))[0]
        if d0 > d1: 
            res =  np.c_[np.linspace(ends[0, 0], ends[1, 0], d0+1, dtype=np.int32),
                        np.round(np.linspace(ends[0, 1], ends[1, 1], d0+1))
                        .astype(np.int32)]
        else:
            res =  np.c_[np.round(np.linspace(ends[0, 0], ends[1, 0], d1+1))
                        .astype(np.int32),
                        np.linspace(ends[0, 1], ends[1, 1], d1+1, dtype=np.int32)]
        return res 

    def connect_torch(self, ends):
        d0, d1 = torch.abs(torch.diff(ends, axis=0))[0]
        if d0 > d1: 
            x = torch.linspace(ends[0, 0], ends[1, 0], d0+1, device = self.device, requires_grad=False )
            y = torch.round(torch.linspace(ends[0, 1], ends[1, 1], d0+1, device = self.device, requires_grad=False))
            res = torch.stack( (x.long(), y.long()), dim=1)
        else:
            x = torch.round(torch.linspace(ends[0, 0], ends[1, 0], d1+1, device = self.device, requires_grad=False))
            y = torch.linspace(ends[0, 1], ends[1, 1], d1+1, device = self.device, requires_grad=False)
            res = torch.stack( (x.long(), y.long()), dim=1)
            
        got_padding = torch.zeros(res.shape[0], 3, device=self.device)
        tea = torch.atan2(ends[1, 0] - ends[1, 1], ends[0, 0] - ends[0, 1])
        got_padding[:,0] = torch.cos(tea)
        got_padding[:,1] = torch.sin(tea)
        got_padding[:,2] = torch.sqrt(  torch.pow(res[:,0] - ends[0, 0],2) +  torch.pow(res[:,1] - ends[0, 1],2) )
        return res.to(self.device), got_padding.to(self.device)

    
    def get_padding_res(self, b, a, start_x, start_y, end_x, end_y, shape, offsets):
        smp = torch.tensor([], device=self.device, dtype=torch.int64)          
        padding_batch_idx, padding_anchor_idx, padding_j, padding_i, padding_got, padding_xy = smp,smp,smp,smp,smp,smp
        for idx in range(b.shape[0]):
            one_b, one_a, one_start_x, one_start_y, one_end_x, one_end_y = b[idx], a[idx], start_x[idx], start_y[idx], end_x[idx], end_y[idx]
            tmp = torch.tensor([ [ one_start_x.long(), one_start_y.long() ],
                                 [ one_end_x.long(),   one_end_y.long()   ]])
            mid_pts, mid_got = self.connect_torch(tmp)
            upsample = 640//shape[3]
            mid_pts, mid_got = mid_pts[upsample:-upsample,:], mid_got[upsample:-upsample,:]
            
            #x&y: bachnorm_output_shape, len: batchorm_1
            mid_pts = mid_pts/torch.tensor([640,640],device=self.device)*torch.tensor([shape[3],shape[2]],device=self.device) #澶氳锛?鍒?
            mid_got[:,2] = mid_got[:,2]/640
            
            one_b = one_b.repeat(mid_pts.shape[0])
            one_a = one_a.repeat(mid_pts.shape[0])
            
            mid_ijs = (mid_pts - offsets).long()
            mid_xy = mid_pts - mid_ijs
            mid_i, mid_j = mid_ijs.T  # grid indices  #锛堝琛岋紝2鍒楋級---->锛?琛岋紝澶氬垪锛?-->涓€缁村拰涓€缁?
            
            padding_batch_idx = torch.cat((padding_batch_idx, one_b), 0)
            padding_anchor_idx = torch.cat((padding_anchor_idx, one_a), 0)
            padding_j = torch.cat((padding_j, mid_j.clamp_(0, shape[2] - 1)), 0)
            padding_i = torch.cat((padding_i, mid_i.clamp_(0, shape[3] - 1)), 0)
            padding_got = torch.cat((padding_got, mid_got), 0)   
            padding_xy = torch.cat((padding_xy, mid_xy), 0)  
        return padding_batch_idx, padding_anchor_idx, padding_j, padding_i, padding_got, padding_xy
    
        
    def build_targets(self, p, targets, open_padding_str):   
        #translation
        #                        A       B       C     D
        #           0      1   2   3   4   5   6   7  8  9
        #targets: img_id  cls  x1  y1  x2  y2  x3  y3 x4 y4
        #print("targets: ", targets)
        theta_AD = torch.atan2(targets[:,9:10] - targets[:,3:4],targets[:,8:9] - targets[:,2:3])
        theta_BC = torch.atan2(targets[:,7:8]  - targets[:,5:6],targets[:,6:7] - targets[:,4:5])
        theta_AB = torch.atan2(targets[:,5:6]  - targets[:,3:4],targets[:,4:5] - targets[:,2:3])
        theta_BA = torch.atan2(targets[:,3:4]  - targets[:,5:6],targets[:,2:3] - targets[:,4:5])

        lengt_AB = torch.sqrt(  torch.pow(targets[:,4:5] - targets[:,2:3],2) +  torch.pow(targets[:,5:6] - targets[:,3:4],2) )
        
        theta_CB = torch.atan2(targets[:,5:6] - targets[:,7:8], targets[:,4:5] - targets[:,6:7])
        theta_DA = torch.atan2(targets[:,3:4] - targets[:,9:10], targets[:,2:3] - targets[:,8:9])
        lengt_AD = torch.sqrt(  torch.pow(targets[:,8:9] - targets[:,2:3],2) +  torch.pow(targets[:,9:10] - targets[:,3:4],2) )
        lengt_BC = torch.sqrt(  torch.pow(targets[:,6:7] - targets[:,4:5],2) +  torch.pow(targets[:,7:8]  - targets[:,5:6],2) )

        tmp = torch.cat((torch.zeros_like(targets,device=self.device),torch.zeros(targets.shape[0],7+10,device=self.device)),dim=1)
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

        tmp[:,17:19] = targets[:,6:8]
        tmp[:,19:20] = torch.cos(theta_CB)
        tmp[:,20:21] = torch.sin(theta_CB)
        tmp[:,21:22] = lengt_BC

        tmp[:,22:24] = targets[:,8:10] 
        tmp[:,24:25] = torch.cos(theta_DA)
        tmp[:,25:26] = torch.sin(theta_DA)
        tmp[:,26:27] = lengt_AD

        #添加一个补丁，由于以前的标注方法有问题，现在统一自车在库位里面，这个库位属于可泊库位
        if (not hyp["ALL_PARKING_LOT_SAME_WEIGHT"]):
            tmp[tmp[:,16]>0.15,1]=0 #车子15%面积在库位里面，以前打的标签是不可泊，现在统一改成可泊库位

        targets=tmp
        #   0      1   2  3   4  5  6    7  8          9  10 11 12  13  14 15     16     17 18  19   20    21   22 23  24   25    26
        #img_id occupy Ax Ay c1 s1  leng c2 s2   &     Bx By c1 s1 leng c2 s2 intersects Cx Cy CBc1 CBs1 CBleng Dx Dy DAc1 DAs1 DAleng


        
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, Atbox, Btbox, Ctbox, Dtbox, Aindices, Bindices, Cindices, Dindices, anch = [], [], [], [], [], [], [], [], [], []
        
        Ctbox_padding, Cindices_padding, Dtbox_padding, Dindices_padding = [], [], [], []
        
        # normalized to gridspace gain: 
        # img_id occupy Ax Ay c1 s1  leng c2 s2 && Bx By c1 s1 leng c2 s2 intersects Cx Cy CBc1 CBs1 CBleng Dx Dy DAc1 DAs1 DAleng arid
        gain = torch.ones(18+10, device=self.device)
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
            gain[17:19] = torch.tensor(shape)[[3, 2]] 
            gain[22:24] = torch.tensor(shape)[[3, 2]] 

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
                tmp_b, tmp_c, tmp_Ax, tmp_Ay, tmp_Ac1, tmp_As1, tmp_Aleng, tmp_Ac2, tmp_As2, tmp_Bx, tmp_By, tmp_Bc1, tmp_Bs1, tmp_Bleng, tmp_Bc2, tmp_Bs2, itst, \
                              tmp_Cx, tmp_Cy, tmp_Cc1, tmp_Cs1, tmp_Cleng, tmp_Dx, tmp_Dy, tmp_Dc1, tmp_Ds1, tmp_Dleng, a = t.chunk(18+10, 1)
                bc = torch.concat((tmp_b,tmp_c), dim=1)
                Agxy = torch.concat((tmp_Ax,tmp_Ay), dim=1)
                Agot = torch.concat((tmp_Ac1,tmp_As1, tmp_Ac2, tmp_As2, tmp_Aleng, itst), dim=1)
                Bgxy = torch.concat((tmp_Bx,tmp_By), dim=1)
                Bgot = torch.concat((tmp_Bc1,tmp_Bs1, tmp_Bc2, tmp_Bs2, tmp_Bleng, itst), dim=1)

                Cgxy = torch.concat((tmp_Cx,tmp_Cy), dim=1)
                Cgot = torch.concat((tmp_Cc1,tmp_Cs1, tmp_Cleng, itst), dim=1)
                Dgxy = torch.concat((tmp_Dx,tmp_Dy), dim=1)
                Dgot = torch.concat((tmp_Dc1,tmp_Ds1, tmp_Dleng, itst), dim=1)

                a, (b, c) = a.long().view(-1), bc.long().T  # anchors, imageID, class
                
                Agij = (Agxy - offsets).long()
                Agi, Agj = Agij.T  # grid indices

                Bgij = (Bgxy - offsets).long()
                Bgi, Bgj = Bgij.T  # grid indices

                Cgij = (Cgxy - offsets).long()
                Cgi, Cgj = Cgij.T  # grid indices

                Dgij = (Dgxy - offsets).long()
                Dgi, Dgj = Dgij.T  # grid indices

                if (open_padding_str):
                    tmp_Ax = tmp_Ax/shape[3]*640
                    tmp_Ay = tmp_Ay/shape[2]*640
                    tmp_Bx = tmp_Bx/shape[3]*640
                    tmp_By = tmp_By/shape[2]*640 
                    tmp_Cx = tmp_Cx/shape[3]*640
                    tmp_Cy = tmp_Cy/shape[2]*640
                    tmp_Dx = tmp_Dx/shape[3]*640
                    tmp_Dy = tmp_Dy/shape[2]*640
                    
                    
                    Cpadding_batch_idx, Cpadding_anchor_idx, Cpadding_j, Cpadding_i, Cpadding_got, Cpadding_xy = self.get_padding_res(b, a, tmp_Bx, tmp_By, tmp_Cx, tmp_Cy, shape, offsets)
                    Cindices_padding.append((Cpadding_batch_idx, Cpadding_anchor_idx, Cpadding_j, Cpadding_i))
                    Ctbox_padding.append(torch.cat((Cpadding_xy, Cpadding_got), 1))
                    
                    Dpadding_batch_idx, Dpadding_anchor_idx, Dpadding_j, Dpadding_i, Dpadding_got, Dpadding_xy = self.get_padding_res(b, a, tmp_Ax, tmp_Ay, tmp_Dx, tmp_Dy, shape, offsets)
                    Dindices_padding.append((Dpadding_batch_idx, Dpadding_anchor_idx, Dpadding_j, Dpadding_i))
                    Dtbox_padding.append(torch.cat((Dpadding_xy, Dpadding_got), 1))
                    
                # Append
                Aindices.append((b, a, Agj.clamp_(0, shape[2] - 1), Agi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
                Bindices.append((b, a, Bgj.clamp_(0, shape[2] - 1), Bgi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
                Cindices.append((b, a, Cgj.clamp_(0, shape[2] - 1), Cgi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
                Dindices.append((b, a, Dgj.clamp_(0, shape[2] - 1), Dgi.clamp_(0, shape[3] - 1)))  # image, anchor, grid

                Atbox.append(torch.cat((Agxy - Agij, Agot), 1))  # box
                Btbox.append(torch.cat((Bgxy - Bgij, Bgot), 1))  # box
                Ctbox.append(torch.cat((Cgxy - Cgij, Cgot), 1))  # box
                Dtbox.append(torch.cat((Dgxy - Dgij, Dgot), 1))  # box

                anch.append(anchors[a])  # anchors
                tcls.append(c)  # class 
        # time.sleep(10)
        # print("tcls: ",tcls)
        # print("Atbox: ",Atbox)
        # print("Aindices: ",Aindices)
        # print("Btbox: ",Btbox)
        # print("Bindices: ",Bindices)
        # print("Ctbox: ",Ctbox)
        # print("Cindices: ",Cindices)
        # print("Dtbox: ",Dtbox)
        # print("Dindices: ",Dindices)
        # print("anch: ",anch)
        # print("Ctbox_padding: ",Ctbox_padding)
        # print("Cindices_padding: ",Cindices_padding)
        # print("Dtbox_padding: ",Dtbox_padding)
        # print("Dindices_padding: ",Dindices_padding)
        # sys.exit()
        return tcls, Atbox, Aindices, Btbox, Bindices, Ctbox, Cindices, Dtbox, Dindices, anch, Ctbox_padding, Cindices_padding, Dtbox_padding, Dindices_padding
