python train.py  --weights xuqing/yolov5s.pt  --cfg ./xuqing/yolov5s_xuqing.yaml  --data ./xuqing/VOC_xuqing.yaml  --device 0,1,2,3,4,5 --imgsz 640 --rect --epoch 600 --batch-size 300  --noautoanchor
