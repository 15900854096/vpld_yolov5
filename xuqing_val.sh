python ./val.py \
--weights ./runs/train/20250620searching/weights/best.pt \
--data ./xuqing/VOC_xuqing.yaml \
--device 0,1,2,3 \
--imgsz 640 \
--batch-size 80 \
--conf-thres 0.45 \
--iou-thres 0.2 \
--half \
--verbose
