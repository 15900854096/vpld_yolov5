python ./val.py --weights ./runs/train/exp3/weights/best.pt --data ./xuqing/VOC_xuqing.yaml --device 0,1,2,3 --imgsz 640 --batch-size 64 --conf-thres 0.45  --iou-thres 0.6 --half --verbose
