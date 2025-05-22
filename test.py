import cv2
from PIL import Image
from ultralytics.models import YOLO

def demo():

    out = model.export(format="engine", imgsz=640, dynamic=True, verbose=False, batch=8, workspace=2, half=True)

    model.info()  # 打印模型信息
    model.summary()  # 打印模型结构

def train():

    # Train the model
    results = model.train(data="./cfg/datasets/BDD100K.yaml", 
                          name="train-yolov8-C2f_TripletAttention-e200",
                          project="ultralytics/runs/detect",
                          epochs=200, 
                          batch=512, 
                          imgsz=640, 
                          device=[0, 1, 2, 3]) # 训练模型


def val():

    # Validate the model
    results = model.val(data="./cfg/datasets/BDD100K.yaml", 
                        batch=16, 
                        imgsz=640, 
                        conf=0.25,
                        iou=0.6,
                        save_json=True,
                        verbose=True,
                        project="ultralytics/runs/detect",
                        name="val-yolov8-e200",
                        device="0") # 验证模型

def predict():

    # Predict with the model
    results = model.predict(source="/home/nrc505/myyolov8/cabc30fc-e7726578.jpg", 
                            conf=0.25, 
                            iou=0.45, 
                            save=True, 
                            save_txt=True, 
                            show=True,
                            visualize=True), # zhanshi zhongjiaceng keshihua


if __name__ == "__main__":

    # model = YOLO("yolov8-CSN.yaml", verbose=True)  # 加载构建yaml自定义模型

    # model = YOLO("yolov8-CGLU.yaml", verbose=True)  # 加载构建预训练模型

    # model = YOLO("yolov8-C2f_TripletAttention.yaml", verbose=True)  # 加载构建预训练模型

    # model = YOLO("yolov8-TripletAttention.yaml", verbose=True)  # 加载构建预训练模型

    model = YOLO("/home/nrc505/myyolov8/ultralytics/runs/detect/train-yolov8-e300/weights/best.pt", verbose=True)  # 加载构建预训练模型

    # model.load("ultralytics/runs/detect/train-yolov8-e200/weights/best.pt")  # 加载预训练模型

    # train()  # 训练模型

    val()  # 验证模型
    
    # predict()  # 预测模型