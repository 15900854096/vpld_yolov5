rm database_train*.cache
python ./val.py \
--weights ./runs/train/20250917parkingDDP22/weights/last_1050.pt  \
--data ./xuqing/VOC_xuqing.yaml \
--device 0,1,2,3,4,5,6,7 \
--imgsz 640 \
--batch-size 40 \
--conf-thres 0.45 \
--iou-thres 0.2 \
--half \
--verbose
