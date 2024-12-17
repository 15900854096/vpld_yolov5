python ./detect.py \
--weights ./runs/train/exp4/weights/last_450.pt \
--source /home/xuqing/ori \
--data ./xuqing/VOC_xuqing.yaml \
--device 3 \
--imgsz 640 \
--conf-thres 0.45 \
--iou-thres 0.2

#/home/xuqing/ori/ \