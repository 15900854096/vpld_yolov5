python ./detect.py \
--weights ./runs/train/20250620searching/weights/best.pt \
--source /home/xuqing/vpld_groundtruth_repair/newrule/parking/yantaishiyanchangdi/40282/images \
--data ./xuqing/VOC_xuqing.yaml \
--device 7 \
--imgsz 640 \
--conf-thres 0.4 \
--iou-thres 0.2

#/home/xuqing/ori/ \