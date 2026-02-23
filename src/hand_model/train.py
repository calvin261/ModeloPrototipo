from ultralytics import YOLO

from .config import TrainConfig


def train_model(cfg: TrainConfig):
    model = YOLO(cfg.weights)
    return model.train(
        data=cfg.data,
        imgsz=cfg.imgsz,
        epochs=cfg.epochs,
        batch=cfg.batch,
        amp=cfg.amp,
        seed=cfg.seed,
        project=cfg.project,
    )
