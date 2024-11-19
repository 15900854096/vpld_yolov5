python ./val.py \
--weights ./runs/train/exp81/weights/last_950.pt \
--data ./xuqing/VOC_xuqing.yaml \
--device 0,1,2,3 \
--imgsz 640 \
--batch-size 80 \
--conf-thres 0.9 \
--iou-thres 0.2 \
--half \
--verbose
