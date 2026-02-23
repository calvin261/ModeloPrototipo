from ultralytics import YOLO


def export_model(weights, format="onnx", imgsz=640, int8=False):
    model = YOLO(weights)
    return model.export(format=format, imgsz=imgsz, int8=int8)
