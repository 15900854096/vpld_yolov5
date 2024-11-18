python ./detect.py \
--weights ./runs/train/20241111/weights/best.pt \
--source /home/xuqing/vpld_groundtruth_repair/yantai/laji \
--data ./xuqing/VOC_xuqing.yaml \
--device 7 \
--imgsz 640 \
--conf-thres 0.4 \
--iou-thres 0.2

#/home/xuqing/ori/ \