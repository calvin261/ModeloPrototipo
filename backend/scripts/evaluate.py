from ultralytics import YOLO


def evaluate(weights, data="datasets/hands.yaml"):
    model = YOLO(weights)
    return model.val(data=data)


def main():
    return None


if __name__ == "__main__":
    main()
