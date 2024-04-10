python ./val.py --weights ./runs/train/exp44/weights/best.pt --data ./xuqing/VOC_xuqing.yaml --device 0,1,2,3,4,5,6,7 --imgsz 640 --batch-size 400 --conf-thres 0.6  --iou-thres 0.1 --half --verbose
