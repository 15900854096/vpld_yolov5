#sleep 7h
python train.py  \
--weights xuqing/yolov5s.pt  \
--cfg ./xuqing/yolov5s_xuqing.yaml  \
--data ./xuqing/VOC_xuqing.yaml  \
--device 0,1,2,3 \
--imgsz 640 \
--rect \
--epoch 1200 \
--batch-size 40 \
--label-smoothing 0.2 \
--noautoanchor \
--patience 0 \
--cache  disk \
--workers 16 \
--sync-bn \
#--resume

#--resume \
# ./runs/train/exp53/weights/best.pt 
#python train.py  --weights xuqing/yolov5s.pt  --cfg ./xuqing/yolov5s_xuqing.yaml  --data ./xuqing/VOC_xuqing.yaml  --device 0,1,2,3,4,5,6,7 --imgsz 640 --rect --epoch 900 --batch-size 400 --label-smoothing 0.2 --noautoanchor --sync-bn

#--sync-bn parkinh use it,searching not use it
#python train.py   --weights /home/xuqing/tools/yolov5_ori/runs/train/exp/weights/best.pt --cfg ./xuqing/yolov5s_xuqing.yaml  --data ./xuqing/VOC_xuqing.yaml  --device 0,1,2,3,4,5,6,7 --imgsz 640 --rect --epoch 900 --batch-size 400  --noautoanchor --sync-bn #--resume 
