python ./detect.py --weights ./runs/train/exp13/weights/best.pt --source /home/xuqing/ps2.0/testing/all --data ./xuqing/VOC_xuqing.yaml --device 0,1,2,3 --imgsz 640 --conf-thres 0.45 --iou-thres 0.65
