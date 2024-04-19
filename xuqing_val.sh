python ./val.py --weights ./runs/train/exp33/weights/best.pt --data ./xuqing/VOC_xuqing.yaml --device 1,3,4,5 --imgsz 640 --batch-size 200 --conf-thres 0.4  --iou-thres 0.2 --half --verbose
