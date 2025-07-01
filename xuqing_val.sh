python ./val.py \
--weights ./runs/train/20250503newrule/weights/last_1199.pt  \
--data ./xuqing/VOC_xuqing.yaml \
--device 0 \
--imgsz 640 \
--batch-size 35 \
--conf-thres 0.9 \
--iou-thres 0.2 \
--half \
--verbose
