python train.py  \
--weights xuqing/yolov5s.pt \
--cfg ./xuqing/yolov5s_xuqing.yaml  \
--data ./xuqing/VOC_xuqing.yaml  \
--device 1 \
--imgsz 640 \
--epoch 300 \
--batch-size 4 \
--label-smoothing 0.2 \
--noautoanchor \
--sync-bn \
--workers 0 \
--patience 0 
#--optimizer Adam
#--resume
#python train.py  --weights xuqing/yolov5s.pt  --cfg ./xuqing/yolov5s_xuqing.yaml  --data ./xuqing/VOC_xuqing.yaml  --device 0,1,2,3,4,5,6,7 --imgsz 640 --rect --epoch 900 --batch-size 400 --label-smoothing 0.2 --noautoanchor --sync-bn
#--rect \
#--sync-bn parkinh use it,searching not use it
#python train.py   --weights /home/xuqing/tools/yolov5_ori/runs/train/exp/weights/best.pt --cfg ./xuqing/yolov5s_xuqing.yaml  --data ./xuqing/VOC_xuqing.yaml  --device 0,1,2,3,4,5,6,7 --imgsz 640 --rect --epoch 900 --batch-size 400  --noautoanchor --sync-bn #--resume 
