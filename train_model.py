from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/train10/weights/last.pt")

    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        resume=False  
    )

if __name__ == "__main__":
    main()
    