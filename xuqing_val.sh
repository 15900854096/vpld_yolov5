python ./val.py \
--weights ./runs/train/exp5/weights/last_900.pt \
--data ./xuqing/VOC_xuqing.yaml \
--device 4,5,6,7 \
--imgsz 640 \
--batch-size 40 \
--conf-thres 0.9 \
--iou-thres 0.2 \
--half \
--verbose
