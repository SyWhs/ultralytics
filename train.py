import cv2
from PIL import Image
from ultralytics.models import YOLO
import torch

def demo():

    out = model.export(format="engine", imgsz=640, dynamic=True, verbose=False, batch=8, workspace=2, half=True)

    model.info()  # 打印模型信息
    model.summary()  # 打印模型结构

def train():

    # Train the model
    results = model.train(data=data,
                          project=project, 
                          name=name,
                          epochs=200, 
                          batch=12,
                          imgsz=640, 
                          device=[0, 1, 2],
                          optimizer="AdamW",
                        #   resume=True,  # 恢复训练
                            ) # 训练模型


def val():

    # Validate the model
    results = model.val(data=data, 
                        project=project,
                        name=name,
                        batch=16, 
                        imgsz=640, 
                        conf=0.25,
                        iou=0.6,
                        save_json=True,
                        verbose=True,
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


    data = "/root/autodl-tmp/ultralytics/ultralytics/cfg/datasets/ExDark.yaml"

    project="/root/autodl-tmp/ultralytics/runs/detect/ExDark/train"

    # name="train-yolov8-raw-e500"

    name="train-yolov8-Retinex-e200-3"

    # model = YOLO("yolov8.yaml", verbose=True)  # 加载预训练模型

    # model = YOLO("yolov8-CSN.yaml", verbose=True)  # 加载构建yaml自定义模型

    # model = YOLO("yolov8-CGLU.yaml", verbose=True)  # 加载构建yaml自定义模型

    # model = YOLO("yolov8-C2f_TripletAttention.yaml", verbose=True)  # 加载构建yaml自定义模型

    # model = YOLO("yolov8-TripletAttention.yaml", verbose=True)  # 加载构建yaml自定义模型

    # model = YOLO("yolov8-CBAM.yaml", verbose=True)  # 加载构建yaml自定义模型

    model = YOLO("yolov8-Retinex.yaml", verbose=True)  # 加载构建yaml自定义模型

    # model = YOLO('/root/autodl-tmp/ultralytics/runs/detect/ExDark/train/train-yolov8-Retinex-e200/weights/last.pt') # 加载中断训练的模型

    # model = YOLO("/home/nrc505/myyolov8/ultralytics/runs/detect/VisDrone/train/train-yolov8-CBAM-e200/weights/best.pt", verbose=True)  # 加载构建训练好的模型

    # model.load("ultralytics/runs/detect/train-yolov8-e200/weights/best.pt")  # 加载预训练模型

    # 1. 加载RetinexFormer权重
    retinex_weights = torch.load('/root/autodl-tmp/ultralytics/pretrain_model/NTIRE.pth')
    # 2. 加载到RetinexFormer层
    model.model.model[0].load_state_dict(retinex_weights['params'])

    # 冻结RetinexFormer参数
    retinex_layer = model.model.model[0]
    for param in retinex_layer.parameters():
        param.requires_grad = False

    train()  # 训练模型

    # val()  # 验证模型
    
    # predict()  # 预测模型
    
    # model.tune(
    #         data=data,
    #         epochs=30,
    #         iterations=300,
    #         optimizer="AdamW",
    #         # space=search_space,
    #         plots=False,
    #         save=False,
    #         val=False,
    #         # device=[0, 1, 2, 3],
    #         )