import json
import os

def process_single_json(json_path, txt_dir, img_width, img_height):
    """处理单个JSON文件"""
    # 类别映射字典
    category_map = {
        "bus": 0,
        "traffic light": 1,
        "traffic sign": 2,
        "person": 3,
        "bike": 4,
        "truck": 5,
        "motor": 6,
        "car": 7,
        "train": 8,
        "rider": 9
    }

    # 生成输出路径
    base_name = os.path.basename(json_path)
    txt_name = os.path.splitext(base_name)[0] + ".txt"
    output_path = os.path.join(txt_dir, txt_name)

    # 读取JSON数据
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 写入TXT文件
    with open(output_path, 'w') as txt_file:
        # 遍历所有帧
        for frame in data.get("frames", []):
            # 遍历每个检测对象
            for obj in frame.get("objects", []):
                # 只处理带有box2d的检测目标
                if "box2d" in obj and obj.get("category") in category_map:
                    # 解析数据
                    class_name = obj["category"]
                    class_id = category_map[class_name]
                    box = obj["box2d"]
                    
                    # 计算归一化坐标
                    x_center = (box["x1"] + box["x2"]) / (2 * img_width)
                    y_center = (box["y1"] + box["y2"]) / (2 * img_height)
                    width = (box["x2"] - box["x1"]) / img_width
                    height = (box["y2"] - box["y1"]) / img_height
                    
                    # 写入文件
                    line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                    txt_file.write(line)

def convert_folder(json_dir, txt_dir, img_width=1280, img_height=720):
    """批量处理整个文件夹"""
    # 创建输出目录
    os.makedirs(txt_dir, exist_ok=True)
    
    # 遍历JSON文件夹
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(json_dir, filename)
            process_single_json(json_path, txt_dir, img_width, img_height)
            print(f"Processed: {filename}")

# 使用示例
if __name__ == "__main__":
    # 原始JSON文件夹路径
    JSON_DIR = "datasets/bdd100k/labels/val"
    # 输出TXT文件夹路径
    TXT_DIR = "datasets/bdd100k/yolo_labels/val"
    
    convert_folder(
        json_dir=JSON_DIR,
        txt_dir=TXT_DIR,
        img_width=1280,
        img_height=720
    )