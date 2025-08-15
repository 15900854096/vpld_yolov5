python ./detect.py \
--weights ./runs/train/20250808searching_select_difficul_neg_sample/weights/last_800.pt \
--source /home/xuqing/vpld_groundtruth_repair/newrule/right_labels/yantiashiyanchangdi_bochuxiangguan_49045/images \
--data ./xuqing/VOC_xuqing.yaml \
--device 0 \
--imgsz 640 \
--conf-thres 0.4 \
--iou-thres 0.2

#/home/xuqing/ori/ \