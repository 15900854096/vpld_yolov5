python ./export.py \
--weights ./runs/train/20250919parkingDDP3/weights/last_150.pt \
--device cpu \
--imgsz 640 \
--include onnx \
--simplify \
--opset 11
