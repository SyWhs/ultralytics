import os
import cv2

if __name__ == '__main__':
    dataDir = "/home/nrc505/myyolov8/datasets/ExDark/images/test"
    saveDir = "/home/nrc505/myyolov8/datasets/ExDark/images/test1"
    
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    for pic in os.listdir(dataDir):
        old_path = os.path.join(dataDir, pic)
        # print("pic:",pic)
        new_img = cv2.imread(old_path)

        # 新增PNG转JPG功能
        if pic.lower().endswith('.png'):  # 包含大小写后缀
            # 修改保存路径的扩展名
            filename = os.path.splitext(pic)[0]  # 分离文件名和扩展名
            new_path = os.path.join(saveDir, f"{filename}.jpg")
        else:
            # 保持原有路径
            new_path = os.path.join(saveDir, pic)

        cv2.imwrite(new_path, new_img)

    print("图片转换成功！")