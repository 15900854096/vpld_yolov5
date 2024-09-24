python ./val.py \
--weights ./runs/train/exp99_20240826/weights/best.pt \
--data ./xuqing/VOC_xuqing.yaml \
--device 0 \
--imgsz 640 \
--batch-size 20 \
--conf-thres 0.4  \
--iou-thres 0.2 \
--half \
--verbose
