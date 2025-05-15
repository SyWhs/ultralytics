import cv2
from PIL import Image
from ultralytics.models import YOLO

def demo():

    out = model.export(format="engine", imgsz=640, dynamic=True, verbose=False, batch=8, workspace=2, half=True)

    model.info()  # 打印模型信息
    model.summary()  # 打印模型结构

    # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
    results = model.predict(source="bus.jpg", show=True)  # Display preds. Accepts all YOLO predict arguments
    for result in results:
        print(result.boxes)

    # from PIL
    im1 = Image.open("bus.jpg")
    results = model.predict(source=im1, save=True)  # save plotted images

    # from ndarray
    im2 = cv2.imread("bus.jpg")
    results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

    # from list of PIL/ndarray
    results = model.predict(source=[im1, im2])

if __name__ == "__main__":

    # model = YOLO("yolov8-CSN.yaml", verbose=True)  # 加载构建yaml自定义模型

    # model = YOLO("yolov8.yaml", verbose=True)  # 加载构建yaml自定义模型

    model = YOLO("yolov8n.pt", verbose=True)  # 加载构建预训练模型

    # model.load("yolov8n.pt")  # 加载预训练模型

    # Train the model
    # results = model.train(data="myyolov8/ultralytics/ultralytics/cfg/datasets/BDD100K.yaml", epochs=200, batch=512, imgsz=640, device=[0, 1, 2, 3]) # 训练模型

    results = model.train(data="/home/nrc505/myyolov8/ultralytics/ultralytics/cfg/datasets/coco8.yaml", epochs=100, imgsz=640) # 训练模型