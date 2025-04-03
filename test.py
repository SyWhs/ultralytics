import cv2
from PIL import Image
from ultralytics import YOLO
def demo():
    model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml", verbose=True)

    out = model.export(format="engine", imgsz=640, dynamic=True, verbose=False, batch=8, workspace=2, half=True)

    model.info()

    # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
    results = model.predict(source="bus.jpg", show=True)  # Display preds. Accepts all YOLO predict arguments
    for result in results:
        print(result.boxes)

    # Train the model
    results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

    # from PIL
    im1 = Image.open("bus.jpg")
    results = model.predict(source=im1, save=True)  # save plotted images

    # from ndarray
    im2 = cv2.imread("bus.jpg")
    results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

    # from list of PIL/ndarray
    results = model.predict(source=[im1, im2])

if __name__ == "__main__":

    model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml", verbose=True)
