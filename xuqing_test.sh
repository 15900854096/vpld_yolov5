python ./detect.py \
--weights ./runs/train/20250718newrule/weights/last_1199.pt \
--source /home/xuqing/16582/images/ \
--data ./xuqing/VOC_xuqing.yaml \
--device 3 \
--imgsz 640 \
--conf-thres 0.45 \
--iou-thres 0.2

#/home/xuqing/ori/ \
