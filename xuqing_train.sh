#sleep 7h
#nice -n -20
rm database_train*.cache
python -m torch.distributed.run \
--nproc_per_node 8 \
train.py  \
--weights xuqing/yolov5s.pt  \
--cfg ./xuqing/yolov5s_xuqing.yaml  \
--data ./xuqing/VOC_xuqing.yaml  \
--name 20250921parkingDDP \
--device 0,1,2,3,4,5,6,7 \
--imgsz 640 \
--epoch 1500 \
--batch-size 512 \
--label-smoothing 0.2 \
--noautoanchor \
--patience 5000 \
--workers 32 \
--sync-bn 
#--freeze 25
#--resume

#
# -m torch.distributed.run \
# --nproc_per_node 8 \
# --master_addr "10.0.8.21" \
# --master_port 1425 \

#--resume \
# ./runs/train/exp53/weights/best.pt 
#python train.py  --weights xuqing/yolov5s.pt  --cfg ./xuqing/yolov5s_xuqing.yaml  --data ./xuqing/VOC_xuqing.yaml  --device 0,1,2,3,4,5,6,7 --imgsz 640 --rect --epoch 900 --batch-size 400 --label-smoothing 0.2 --noautoanchor --sync-bn

#--sync-bn parkinh use it,searching not use it
#python train.py   --weights /home/xuqing/tools/yolov5_ori/runs/train/exp/weights/best.pt --cfg ./xuqing/yolov5s_xuqing.yaml  --data ./xuqing/VOC_xuqing.yaml  --device 0,1,2,3,4,5,6,7 --imgsz 640 --rect --epoch 900 --batch-size 400  --noautoanchor --sync-bn #--resume 
