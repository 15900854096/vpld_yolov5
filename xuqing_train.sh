python train.py  --weights xuqing/yolov5s.pt  --cfg ./xuqing/yolov5s_xuqing.yaml  --data ./xuqing/VOC_xuqing.yaml  --device 0,1,2,3,4,5,6,7 --imgsz 640 --rect --epoch 300 --batch-size 256 --noautoanchor
