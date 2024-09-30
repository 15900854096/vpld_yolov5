import cv2
import cv2 as cv
import numpy as np
import sys
import copy
import torch
import os

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
sscnt=0
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


def batch_draw_arr_save(imgs,arress):
    imgs = np.array(imgs)
    n, c, h, w = imgs.shape
    arress = np.array(arress)
    global sscnt
    for  i,img in enumerate(imgs):
        img_ = img.copy()
        img_ = img_.transpose((1, 2, 0))[:,:,::-1]
        img_ = np.ascontiguousarray(img_)
        arrs = arress[arress[:,0] == i]
        
        for arr in arrs:
            p1 = (round(float(arr[2]*w)) , round(float(arr[3]*h)))
            p2 = (round(float(arr[4]*w)) , round(float(arr[5]*h)))
            
            cv.arrowedLine(img_, p1, p2, point_color, thickness, lineType)
            if(arr[6]!=-1):
                p3 = (round(float(arr[6]*w)) , round(float(arr[7]*h)))
                cv.arrowedLine(img_, p2, p3, point_color, thickness, lineType)      
                
        cv2.imwrite(r"./xuqing/flipud/ss_%06d.jpg"%sscnt, img_)
        print("save img to ",r"./xuqing/flipud/ss_%06d.jpg"%sscnt)
        sscnt+=1
        if(sscnt>100):
            sys.exit()
            

def batch_draw_everything_save(imgs, labelses, masks, arress, paths):
    global sscnt
    
    imgs = np.array(imgs) #nchw grb
    labelses = np.array(labelses)
    masks = np.array(masks) #n1hw
    arress = np.array(arress)
    
    n, c, h, w = imgs.shape
    
    for idx in range(n):
        img    = imgs[idx]
        mask   = masks[idx]
        labels = labelses[labelses[:,0] == idx]
        arres  = arress[arress[:,0] == idx]
        path = paths[idx] 
        
        img = img.transpose((1, 2, 0))[:,:,::-1] #chw rgb-> hwc rgb ->hwc bgr
        
        #draw fs
        mask = torch.squeeze(torch.from_numpy(mask)).long()
        pre_color = copy.deepcopy(img) #hwc bgr
        bchanel =  np.vectorize(dict_col_b.get)(mask)
        gchanel =  np.vectorize(dict_col_g.get)(mask)
        rchanel =  np.vectorize(dict_col_r.get)(mask)
        pre_color[:,:,0], pre_color[:,:,1], pre_color[:,:,2]  = bchanel, gchanel, rchanel
        img = cv2.addWeighted(pre_color,0.3,img,0.7,0)

        #draw vpld
        for ele in labels:
            p1 = (round(float(ele[2]*w)) , round(float(ele[3]*h)))
            p2 = (round(float(ele[4]*w)) , round(float(ele[5]*h)))
            p3 = (round(float(ele[6]*w)) , round(float(ele[7]*h)))
            cv.arrowedLine(img, p1, p2, point_color, thickness, lineType)
            cv.arrowedLine(img, p2, p3, point_color, thickness, lineType)
           
        #draw arr
        for arr in arres:
            p1 = (round(float(arr[2]*w)) , round(float(arr[3]*h)))
            p2 = (round(float(arr[4]*w)) , round(float(arr[5]*h)))
            cv.arrowedLine(img, p1, p2, point_color, thickness, lineType)
            if(arr[1]==0):
                p3 = (round(float(arr[6]*w)) , round(float(arr[7]*h)))
                cv.arrowedLine(img, p2, p3, point_color, thickness, lineType)  
        
        name = os.path.basename(path) 
        jobid = os.path.basename(os.path.dirname(os.path.dirname(path)))
        dest = r"./xuqing/flipud/%s_%s_%06d.jpg"%(jobid,name,sscnt)
        cv2.imwrite(dest, img)
        print("save img to ", dest)
        sscnt+=1
        if(sscnt>100):
            sys.exit()