#sleep 7h
python -m torch.distributed.run \
--nproc_per_node 8 \
--master_addr "10.0.8.21" \
--master_port 1422 \
train.py \
--weights xuqing/yolov5s.pt \
--name 20250815searching_DDP \
--cfg ./xuqing/yolov5s_xuqing.yaml  \
--data ./xuqing/VOC_xuqing.yaml  \
--device 0,1,2,3,4,5,6,7 \
--imgsz 640 \
--epoch 900 \
--batch-size 320 \
--label-smoothing 0.2 \
--noautoanchor \
--patience 0 \
--cache  disk \
--workers 32 \
--sync-bn \
#--resume

#--resume \
# ./runs/train/exp53/weights/best.pt 
#python train.py  --weights xuqing/yolov5s.pt  --cfg ./xuqing/yolov5s_xuqing.yaml  --data ./xuqing/VOC_xuqing.yaml  --device 0,1,2,3,4,5,6,7 --imgsz 640 --rect --epoch 900 --batch-size 400 --label-smoothing 0.2 --noautoanchor --sync-bn

#--sync-bn parkinh use it,searching not use it
#python train.py   --weights /home/xuqing/tools/yolov5_ori/runs/train/exp/weights/best.pt --cfg ./xuqing/yolov5s_xuqing.yaml  --data ./xuqing/VOC_xuqing.yaml  --device 0,1,2,3,4,5,6,7 --imgsz 640 --rect --epoch 900 --batch-size 400  --noautoanchor --sync-bn #--resume 
