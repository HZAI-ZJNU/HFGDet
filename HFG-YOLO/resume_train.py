from ultralytics import YOLO
import logging

if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        level=logging.INFO
    )
    logging.info("Training started.")
    model = YOLO("visdrone/train/weights/last.pt")
    results = model.train(
        data="VisDrone.yaml", epochs=350, imgsz=640, batch=16, device=[0],project="visdrone/",
        resume=True
    ) 
    metrics = model.val()  # evaluate model performance on the validation set
    logging.info("Training finished.")
