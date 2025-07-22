import cv2
import cv2 as cv
import numpy as np
import sys
import os
from pathlib import Path

from utils.general import scale_boxes

point_color = [(0, 255, 0), #绿色
               (0, 0, 255), #红色
               (255, 0, 0)] #蓝色 BGR
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
        cv.arrowedLine(img_, p1, p2, point_color[int(ele[0])], thickness, lineType)
        cv.arrowedLine(img_, p2, p3, point_color[int(ele[0])], thickness, lineType)   
    cv2.imwrite(address, img_)
    print("save img to ",address)
    

cnt=0
def batch_draw_save(imgs,labelses):
    imgs = np.array(imgs)
    labelses = np.array(labelses)
    global cnt
    for  i,img in enumerate(imgs):
        img=img[::-1]
        img = img.transpose((1, 2, 0))
        idx = labelses[:,0] == i
        labels = labelses[idx]
        draw_save(img,labels[:,1:], os.path.join(Path.cwd(), r"xuqing/flipud/%06d.jpg"%cnt))
        cnt+=1
        if(cnt>200):
            sys.exit()
        
def draw_points(img, pts, trainshape, rad, color):
    if len(pts):
        pts[:,:2] = scale_boxes(trainshape, pts[:, :2], img.shape).round()
        for x,y,cosv,sinv,lengv,cobj in reversed(pts):
            h,w,c = img.shape
            lengv *=w
            cpoint1 = (round(float(x)) , round(float(y)))
            cpoint2 = (round(float(x+lengv*cosv)) , round(float(y+lengv*sinv)))
            cv2.circle(img, cpoint1, rad, color, 4)
            #cv2.arrowedLine(img, cpoint1, cpoint2, color, 1, 4)    