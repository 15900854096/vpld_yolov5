python ./val.py \
--weights ./runs/train/exp81/weights/last_1150.pt \
--data ./xuqing/VOC_xuqing.yaml \
--device 1,2,3 \
--imgsz 640 \
--batch-size 30 \
--conf-thres 0.9 \
--iou-thres 0.2 \
--half \
--verbose
