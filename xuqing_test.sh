python ./detect.py \
--weights ./runs/train/20241127/weights/best.pt \
--source /home/xuqing/ori \
--data ./xuqing/VOC_xuqing.yaml \
--device 7 \
--imgsz 640 \
--conf-thres 0.4 \
--iou-thres 0.2

#/home/xuqing/ori/ \