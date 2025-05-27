import os
import shutil
import re
from pathlib import Path

# 类别名称映射（数字 -> 目录名）
CLASS_MAPPING = {
    1: "Bicycle",
    2: "Boat",
    3: "Bottle",
    4: "Bus",
    5: "Car",
    6: "Cat",
    7: "Chair",
    8: "Cup",
    9: "Dog",
    10: "Motorbike",
    11: "People",
    12: "Table"
}

def parse_split_info(annotation_file):
    """解析标注文件，自动跳过首行标题并处理多种分隔符"""
    split_info = {}
    with open(annotation_file, 'r') as f:
        # 跳过首行标题
        next(f)
        
        for line_num, line in enumerate(f, start=2):
            # 清理分隔符并分割列
            cleaned_line = re.sub(r'[|\t]+', ' ', line.strip())
            parts = cleaned_line.split()
            
            if len(parts) < 5:
                print(f"跳过第{line_num}行（列数不足）：{line.strip()}")
                continue
                
            try:
                img_name = parts[0].strip()
                class_num = int(parts[1])
                split_code = parts[4]
                
                # 验证类别有效性
                if class_num not in CLASS_MAPPING:
                    print(f"第{line_num}行无效类别：{class_num}")
                    continue
                
                split_info[img_name] = {
                    'class_name': CLASS_MAPPING[class_num],
                    'split_code': split_code
                }
                
            except ValueError as e:
                print(f"解析第{line_num}行失败：{str(e)}")
                print(f"问题行内容：{line.strip()}")
    
    return split_info

def organize_dataset(split_info, img_root, label_root, output_root):
    """执行文件整理，增强错误处理"""
    # 创建目标目录
    for split in ['train', 'val', 'test']:
        Path(os.path.join(output_root, 'images', split)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(output_root, 'labels', split)).mkdir(parents=True, exist_ok=True)

    # 遍历所有图片类别目录
    for class_name in os.listdir(img_root):
        class_img_dir = os.path.join(img_root, class_name)
        class_label_dir = os.path.join(label_root, class_name)
        
        # 跳过非目录文件
        if not os.path.isdir(class_img_dir):
            continue
            
        # 处理每个图片文件
        for img_file in os.listdir(class_img_dir):
            try:
                # 提取基础文件名（兼容不同扩展名）
                base_name = Path(img_file).stem
                img_ext = Path(img_file).suffix
                
                # 尝试匹配可能的文件名格式
                possible_keys = [
                    f"{base_name}{img_ext}",  # 原始文件名
                    f"{base_name}.jpg",       # 可能的大小写变化
                    f"{base_name}.png",
                    base_name                # 无扩展名情况
                ]
                
                # 查找匹配的键
                info = None
                for key in possible_keys:
                    if key in split_info:
                        info = split_info[key]
                        break
                
                if not info:
                    print(f"缺少分割信息：{base_name}")
                    continue
                    
                # 验证类别一致性
                if info['class_name'] != class_name:
                    print(f"类别不匹配：{img_file} 应属于 {info['class_name']} 但存放在 {class_name} 目录")
                    continue
                
                # 转换分割代码
                split_map = {'1': 'train', '2': 'val', '3': 'test'}
                target_split = split_map.get(info['split_code'], 'unknown')
                if target_split == 'unknown':
                    print(f"无效的分割代码：{info['split_code']}（文件：{img_file}）")
                    continue
                
                # 构建路径
                src_img = os.path.join(class_img_dir, img_file)
                src_label = os.path.join(class_label_dir, f"{base_name}.txt")
                
                dst_img = os.path.join(output_root, 'images', target_split, img_file)
                dst_label = os.path.join(output_root, 'labels', target_split, f"{base_name}.txt")
                
                # 复制文件（带错误检查）
                if os.path.exists(src_img):
                    shutil.copy2(src_img, dst_img)
                else:
                    print(f"图片文件缺失：{src_img}")
                    
                if os.path.exists(src_label):
                    shutil.copy2(src_label, dst_label)
                else:
                    print(f"标签文件缺失：{src_label}")
                    
            except Exception as e:
                print(f"处理文件 {img_file} 时发生错误：{str(e)}")
                continue

if __name__ == "__main__":
    # 配置路径
    config = {
        "annotation_file": "/home/nrc505/myyolov8/datasets/EXDark/imageclasslist.txt",
        "img_root": "/home/nrc505/myyolov8/datasets/EXDark/ExDark/ExDark",
        "label_root": "/home/nrc505/myyolov8/datasets/EXDark/labels",
        "output_root": "/home/nrc505/myyolov8/datasets/ExDark"
    }

    # 解析标注文件
    print("开始解析标注文件...")
    split_info = parse_split_info(config["annotation_file"])
    print(f"成功解析 {len(split_info)} 条有效记录")
    
    # 整理数据集
    print("\n开始整理数据集...")
    organize_dataset(
        split_info=split_info,
        img_root=config["img_root"],
        label_root=config["label_root"],
        output_root=config["output_root"]
    )
    print("\n数据集整理完成！")