python ./val.py --weights ./runs/train/exp13/weights/best.pt --data ./xuqing/VOC_xuqing.yaml --device 0,1,2,3 --imgsz 640 --batch-size 32 --conf-thres 0.45  --iou-thres 0.6 --half --verbose
