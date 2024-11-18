python ./export.py \
--weights ./runs/train/exp70/weights/last_0.pt \
--device cpu \
--imgsz 640 \
--include onnx \
--simplify \
--opset 11
