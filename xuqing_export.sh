python ./export.py \
--weights ./runs/train/20250627newrule/weights/last_1199.pt \
--device cpu \
--imgsz 640 \
--include onnx \
--simplify \
--opset 11
