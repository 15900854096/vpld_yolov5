python ./detect.py \
--weights ./runs/train/20250817searching_DDP/weights/best.pt \
--source /home/xuqing/ori \
--data ./xuqing/VOC_xuqing.yaml \
--device 0 \
--imgsz 640 \
--conf-thres 0.4 \
--iou-thres 0.2

#/home/xuqing/ori/ \