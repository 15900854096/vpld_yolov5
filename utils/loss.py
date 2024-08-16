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
from pathlib import Path
from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel

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
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        MSEwh = nn.MSELoss()
        MSEthetaAD = nn.MSELoss(reduction='none')
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
        self.MSEwh, self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = MSEwh, BCEcls, BCEobj, 1.0, h, autobalance
        self.fslossF =nn.MSELoss(reduction='sum') # MultiClassDiceLoss() #nn.MSELoss(reduction='sum')
        self.MSEobj = MSEobj
        self.MSEthetaAD = MSEthetaAD
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.fs_num_class = h['fs_num_class']
        self.device = device

    def __call__(self, p, targets, masks):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        lfs  = torch.zeros(1, device=self.device)  # object loss
        #tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        tcls, Atbox, Aindices, Btbox, Bindices, anchors = self.build_targets(p["vpld"], targets)
        
        vpld_weight = hyp["vpld_weight"]
        fs_weight = hyp["fs_weight"]
        if hyp["task_fs"]:
            fs_gt = self.build_fs_targets(masks)
            lfs += self.fslossF(p["fs"][0], fs_gt) * fs_weight #self.BCEcls(pcls, t)  # BCE
            #lfs += self.fslossF(p["fs"][0], fs_gt) * fs_weight
            
        random.seed(time.time_ns()%(2**32 - 1))
        # Losses
        for i, pi in enumerate(p["vpld"]):  # layer index, layer predictions
            Ab, Aa, Agj, Agi = Aindices[i]  # image, anchor, gridy, gridx
            Bb, Ba, Bgj, Bgi = Bindices[i]  # image, anchor, gridy, gridx

            Atobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj
            Btobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            #Atobj = torch.full_like(Atobj, self.cn, device=self.device)
            #Btobj = torch.full_like(Btobj, self.cn, device=self.device)
            
            na = Ab.shape[0]
            nb = Bb.shape[0]
            n = na + nb  # number of targets
            if n:
                #                                    0  1  2   3    4   5   6    7    8  9  10  11   12  13  14   15   16   17
                #target-subset of predictions #pred: Ax Ay Ac1 As1 Ac2 As2 Alen Aobj  Bx By Bc1 Bs1  Bc2 Bs2 Blen Bobj cls1 cls2
                Apxy, Apot, Aobj, _, _    = pi[Ab, Aa, Agj, Agi].split((2, 5, 1, 8, self.nc), 1)  
                _, Bpxy, Bpot, Bobj, pcls = pi[Bb, Ba, Bgj, Bgi].split((8, 2, 5, 1, self.nc), 1) 

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
                
                
                lossxy  = self.MSEwh(Apbox[:,0:2], Atbox[i][:,0:2]) + self.MSEwh(Bpbox[:,0:2], Btbox[i][:,0:2])
                losstheAD =  torch.sum(self.MSEthetaAD(Apbox[:,2:4], Atbox[i][:,2:4]) * AweightstheAD) / (na*2)+  torch.sum(self.MSEthetaAD(Bpbox[:,2:4], Btbox[i][:,2:4]) * BweightstheAD) / (nb*2)
                losstheAB = self.MSEwh(Apbox[:,4:6], Atbox[i][:,4:6]) + self.MSEwh(Bpbox[:,4:6], Btbox[i][:,4:6])
                losslen = self.MSEwh(Apbox[:,6:7], Atbox[i][:,6:7]) + self.MSEwh(Bpbox[:,6:7], Btbox[i][:,6:7])
                
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
                    lcls += self.MSEwh(pcls.sigmoid(), t) #self.BCEcls(pcls, t)  # BCE

            if na:
                Aselect = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)
                Aselect[Ab, Aa, Agj, Agi] = 1
                for idx, v in enumerate(Ab):
                    list1 = [random.randint(0,pi.shape[2]-1) for j in range(int(hyp["NEG_POS_RATE"]))]
                    list2 = [random.randint(0,pi.shape[3]-1) for j in range(int(hyp["NEG_POS_RATE"]))]
                    Aselect[v,Aa[idx],list1,list2] = 1  
            else:
                Aselect = torch.ones(pi.shape[:4], dtype=pi.dtype, device=self.device)
            
            if nb:
                Bselect = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)
                Bselect[Bb, Ba, Bgj, Bgi] = 1
                for idx, v in enumerate(Bb):
                    list1 = [random.randint(0,pi.shape[2]-1) for j in range(int(hyp["NEG_POS_RATE"]))]
                    list2 = [random.randint(0,pi.shape[3]-1) for j in range(int(hyp["NEG_POS_RATE"]))]
                    Bselect[v,Ba[idx],list1,list2] = 1  
            else:
                Bselect = torch.ones(pi.shape[:4], dtype=pi.dtype, device=self.device)
            
            #Aobji = self.MSEwh(pi[..., 7].sigmoid(), Atobj) #self.BCEobj(pi[..., 7], Atobj)
            #Bobji = self.MSEwh(pi[..., 15].sigmoid(), Btobj) #self.BCEobj(pi[..., 15], Btobj)
            Aobji = torch.sum(self.MSEobj(pi[..., 7].sigmoid(), Atobj) * Aselect) / torch.sum(Aselect)
            Bobji = torch.sum(self.MSEobj(pi[..., 15].sigmoid(), Btobj) * Bselect) / torch.sum(Bselect)
            
            lobj += (Aobji+Bobji) * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box'] * vpld_weight
        lobj *= self.hyp['obj'] * vpld_weight
        lcls *= self.hyp['cls'] * vpld_weight
        bs = Atobj.shape[0]+Btobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs+lfs, torch.cat((lbox, lobj, lcls, lfs)).detach()

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

        tmp = torch.cat((torch.zeros_like(targets,device=self.device),torch.zeros(targets.shape[0],6,device=self.device)),dim=1)
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
        targets=tmp
        #   0      1   2  3   4  5  6    7  8          9  10 11 12  13  14 15
        #img_id occupy Ax Ay c1 s1  leng c2 s2   &     Bx By c1 s1 leng c2 s2


        
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, Atbox, Btbox, Aindices, Bindices, anch = [], [], [], [], [], []

        # normalized to gridspace gain: img_id occupy Ax Ay c1 s1  leng c2 s2 && Bx By c1 s1 leng c2 s2 arid
        gain = torch.ones(17, device=self.device)
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
                    
                if hyp["USE_THREE_POSITIVE_SAMPLE"]:
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
                tmp_b, tmp_c, tmp_Ax,tmp_Ay,tmp_Ac1, tmp_As1, tmp_Aleng, tmp_Ac2, tmp_As2, tmp_Bx, tmp_By, tmp_Bc1, tmp_Bs1, tmp_Bleng, tmp_Bc2, tmp_Bs2, a = t.chunk(17, 1)
                bc = torch.concat((tmp_b,tmp_c), dim=1)
                Agxy = torch.concat((tmp_Ax,tmp_Ay), dim=1)
                Agot = torch.concat((tmp_Ac1,tmp_As1, tmp_Ac2, tmp_As2, tmp_Aleng), dim=1)
                Bgxy = torch.concat((tmp_Bx,tmp_By), dim=1)
                Bgot = torch.concat((tmp_Bc1,tmp_Bs1, tmp_Bc2, tmp_Bs2, tmp_Bleng), dim=1)
                
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
    
    def build_fs_targets(self, masks):
        bs, c, h, w=masks.shape
        fs_gt = torch.zeros((bs, self.fs_num_class, h, w), dtype=masks.dtype, device=self.device)
        for i in range(self.fs_num_class):
            fs_gt[:,i:i+1] = masks[:]==i
        return fs_gt.to(torch.float32)

class BinaryDiceLoss(nn.Module):
	def __init__(self):
		super(BinaryDiceLoss, self).__init__()
	
	def forward(self, input, targets):
		# 获取每个批次的大小 N
		N = targets.size()[0]
		# 平滑变量
		smooth = 1
		# 将宽高 reshape 到同一纬度
		input_flat = input.view(N, -1)
		targets_flat = targets.view(N, -1)
	
		# 计算交集
		intersection = input_flat * targets_flat 
		N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
		# 计算一个批次中平均每张图的损失
		loss = 1 - N_dice_eff.sum() / N
		return loss

class MultiClassDiceLoss(nn.Module):
	def __init__(self, weight=None, ignore_index=None, **kwargs):
		super(MultiClassDiceLoss, self).__init__()
		self.weight = weight
		self.ignore_index = ignore_index
		self.kwargs = kwargs
	
	def forward(self, input, target):
		"""
			input tesor of shape = (N, C, H, W)
			target tensor of shape = (N, H, W)
		"""
		# 先将 target 进行 one-hot 处理，转换为 (N, C, H, W)
		nclass = input.shape[1]
		#target = one_hot(target.long(), nclass)

		assert input.shape == target.shape, "predict & target shape do not match"
		
		binaryDiceLoss = BinaryDiceLoss()
		total_loss = 0
		
		# 归一化输出
		#logits = F.softmax(input, dim=1)
		C = target.shape[1]
		
		# 遍历 channel，得到每个类别的二分类 DiceLoss
		for i in range(C):
			dice_loss = binaryDiceLoss(input[:, i], target[:, i])
			total_loss += dice_loss
		
		# 每个类别的平均 dice_loss
		return total_loss / C
