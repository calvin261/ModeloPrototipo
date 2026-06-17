from ultralytics import YOLO


def export(weights, format="onnx", imgsz=640, int8=False):
    model = YOLO(weights)
    return model.export(format=format, imgsz=imgsz, int8=int8)


def main():
    return None


if __name__ == "__main__":
    main()
