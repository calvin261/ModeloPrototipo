from ultralytics import YOLO


def train(
    weights="models/pretrained/yolov8n.pt",
    data="datasets/hands.yaml",
    imgsz=640,
    epochs=100,
    batch=-1,
    amp=True,
    seed=42,
    project="models/finetuned",
):
    model = YOLO(weights)
    return model.train(
        data=data,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        amp=amp,
        seed=seed,
        project=project,
    )


def main():
    train()


if __name__ == "__main__":
    main()
