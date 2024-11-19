python ./detect.py \
--weights ./runs/train/exp81/weights/last_1150.pt \
--source /home/xuqing/vpld_groundtruth_repair/public_Boden/AVM_000000_001999/images \
--data ./xuqing/VOC_xuqing.yaml \
--device 7 \
--imgsz 640 \
--conf-thres 0.9 \
--iou-thres 0.2

#/home/xuqing/ori/ \