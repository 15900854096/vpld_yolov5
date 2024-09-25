import cv2
import cv2 as cv
import numpy as np
import sys
import copy
import torch

dict_col={
    0:[255,0,0],
    1:[0,0,255],
    2:[0,255,0],
    3:[125,0,0],
    4:[0,125,0],
    5:[0,0,125],
    6:[125,125,0],
    7:[125,0,125],
    8:[0,125,125],
    9:[125,125,125],
}

dict_col_b = {x: dict_col[x][0] for x in dict_col.keys()}
dict_col_g = {x: dict_col[x][1] for x in dict_col.keys()}
dict_col_r = {x: dict_col[x][2] for x in dict_col.keys()}

list_col = list(dict_col.values())

point_color = (0, 255, 0)  # BGR
thickness = 1
lineType = 8

def draw_save(img,labels,address):
    img_ = img.copy()
    h,w,c = img_.shape
    for idx in range(labels.shape[0]):
        ele=labels[idx]
        p1 = (round(float(ele[1]*w)) , round(float(ele[2]*h)))
        p2 = (round(float(ele[3]*w)) , round(float(ele[4]*h)))
        p3 = (round(float(ele[5]*w)) , round(float(ele[6]*h)))
        cv.arrowedLine(img_, p1, p2, point_color, thickness, lineType)
        cv.arrowedLine(img_, p2, p3, point_color, thickness, lineType)   
    cv2.imwrite(address, img_)
    print("save img to ",address)
    

vpldcnt=0
fscnt=0
def batch_draw_save(imgs,labelses):
    imgs = np.array(imgs)
    labelses = np.array(labelses)
    global vpldcnt
    for  i,img in enumerate(imgs):
        img=img[::-1]
        img = img.transpose((1, 2, 0))
        idx = labelses[:,0] == i
        labels = labelses[idx]
        draw_save(img,labels[:,1:],r"./xuqing/flipud/vpld_%06d.jpg"%vpldcnt)
        vpldcnt+=1
        if(vpldcnt>1000):
            sys.exit()
            
def batch_draw_mask_save(imgs, masks):
    imgs = np.array(imgs) #nchw grb
    masks = np.array(masks) #n1hw
    global fscnt
    for img,mask in zip(imgs,masks):
        mask = torch.squeeze(torch.from_numpy(mask)).long()
        img = img.transpose((1, 2, 0))[:,:,::-1] #chw rgb-> hwc rgb ->hwc bgr
        pre_color = copy.deepcopy(img) #hwc bgr
        
        bchanel =  np.vectorize(dict_col_b.get)(mask)
        gchanel =  np.vectorize(dict_col_g.get)(mask)
        rchanel =  np.vectorize(dict_col_r.get)(mask)
        
        pre_color[:,:,0], pre_color[:,:,1], pre_color[:,:,2]  = bchanel, gchanel, rchanel
        xuanran_img = cv2.addWeighted(pre_color,0.3,img,0.7,0)
        cv2.imwrite(r"./xuqing/flipud/freespace_%06d.jpg"%fscnt,xuanran_img)
        fscnt+=1
        if(fscnt>200):
            sys.exit()                  