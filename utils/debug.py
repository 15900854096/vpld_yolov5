import cv2
import cv2 as cv
import numpy as np
import sys


point_color = (0, 255, 0)  # BGR
thickness = 2
lineType = 4

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
        draw_save(img,labels[:,1:],r"/home/xuqing/tools/yolov5_ori/xuqing/flipud/%06d.jpg"%cnt)
        cnt+=1
        if(cnt>300):
            sys.exit()
        