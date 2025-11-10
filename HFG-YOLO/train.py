from ultralytics import YOLO
import logging

if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        level=logging.INFO
    )
    logging.info("Training started.")
    # Load a model
    model = YOLO("yolo11n-hfg.yaml")
    # model = YOLO("visdrone/yolov11n/18/train2/weights/last.pt")
    results = model.train(
        data="VisDrone.yaml", epochs=350, imgsz=640, batch=16, 
        project="visdrone/",
        device=[0],
        # resume=True
    )
    metrics = model.val()  # evaluate model performance on the validation set
    logging.info("Training finished.")
