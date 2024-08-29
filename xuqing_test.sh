python ./detect.py \
--weights ./runs/train/exp/weights/last_1100.pt \
--source /home/xuqing/vpld_groundtruth_repair/123/images \
--data ./xuqing/VOC_xuqing.yaml \
--device 4 \
--imgsz 640 \
--conf-thres 0.4 \
--iou-thres 0.2
