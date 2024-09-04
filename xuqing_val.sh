python ./val.py \
--weights ./runs/train/exp112/weights/best.pt \
--data ./xuqing/VOC_xuqing.yaml \
--device 7 \
--imgsz 640 \
--batch-size 1 \
--conf-thres 0.4  \
--iou-thres 0.2 \
--half \
--verbose
