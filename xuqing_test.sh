python ./detect.py \
--weights ./runs/train/exp99_20240826/weights/best.pt \
--source /home/xuqing/vpld_groundtruth_repair/longmao/917_1310_shanghai_alizhongxin_16879_ok/images \
--data ./xuqing/VOC_xuqing.yaml \
--device 5 \
--imgsz 640 \
--conf-thres 0.4 \
--iou-thres 0.2
