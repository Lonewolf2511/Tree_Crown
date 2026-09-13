from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")

    results = model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        name="tree_crown_yolo11"
    )

    print("Training completed!")

if __name__ == "__main__":
    main()