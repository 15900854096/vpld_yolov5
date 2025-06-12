python ./detect.py \
--weights ./runs/train/20250425newrule/weights/last_1199.pt \
--source /home/xuqing/ori \
--data ./xuqing/VOC_xuqing.yaml \
--device 3 \
--imgsz 640 \
--conf-thres 0.45 \
--iou-thres 0.2

#/home/xuqing/ori/ \
