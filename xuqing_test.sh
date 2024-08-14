python ./detect.py \
--weights ./runs/train/exp52/weights/best.pt \
--source /home/xuqing/vpld_groundtruth_repair/public_Boden/AVM_000000_001999/images \
--data ./xuqing/VOC_xuqing.yaml \
--device 0 \
--imgsz 640 \
--conf-thres 0.4 \
--iou-thres 0.2