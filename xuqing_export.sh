python ./export.py \
--weights ./runs/train/20250917parkingDDP22/weights/last_1050.pt \
--device cpu \
--imgsz 640 \
--include onnx \
--simplify \
--opset 11
