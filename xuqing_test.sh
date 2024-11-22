python ./detect.py \
--weights ./runs/train/exp5/weights/last_800.pt \
--source /home/xuqing/vpld_groundtruth_repair/public_Boden/AVM_000000_001999/images \
--data ./xuqing/VOC_xuqing.yaml \
--device 3 \
--imgsz 640 \
--conf-thres 0.45 \
--iou-thres 0.2

#/home/xuqing/ori/ \