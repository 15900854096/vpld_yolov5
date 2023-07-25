conda activate yolov5

python ./val.py --weights ./runs/train/onePositive/weights/last_299.pt --data ./xuqing/VOC_xuqing.yaml --device 0,1,2,3 --imgsz 640 --batch-size 64 --conf-thres 0.45  --iou-thres 0.6 --verbose
