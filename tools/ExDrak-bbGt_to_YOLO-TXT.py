import os
from PIL import Image

class_mapping = {
    "Bicycle": 0,
    "Boat": 1,
    "Bottle": 2,
    "Bus": 3,
    "Car": 4,
    "Cat": 5,
    "Chair": 6,
    "Cup": 7,
    "Dog": 8,
    "Motorbike": 9,
    "People": 10,
    "Table": 11
}

def find_image_file(img_root, class_name, base_name):
    """在图片目录中查找匹配的文件"""
    class_img_dir = os.path.join(img_root, class_name)
    if not os.path.exists(class_img_dir):
        return None
    
    # 尝试不同扩展名
    extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    for ext in extensions:
        img_path = os.path.join(class_img_dir, f"{base_name}{ext}")
        if os.path.exists(img_path):
            return img_path
    return None

def convert_bbgt_to_yolo(ann_root, img_root, output_root):
    # 遍历标注根目录下的所有类别文件夹
    for class_name in os.listdir(ann_root):
        class_ann_dir = os.path.join(ann_root, class_name)
        if not os.path.isdir(class_ann_dir):
            continue
        
        # 创建对应的输出目录
        class_output_dir = os.path.join(output_root, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        
        # 处理每个标注文件
        for ann_file in os.listdir(class_ann_dir):
            if not ann_file.endswith('.txt'):
                continue
            
            # 解析基础文件名
            base_name = os.path.splitext(ann_file)[0]  # 去除.txt后缀
            if '.' in base_name:  # 处理类似2015_00001.png.txt的情况
                base_name = os.path.splitext(base_name)[0]
            
            # 查找对应的图片文件
            img_path = find_image_file(img_root, class_name, base_name)
            if not img_path:
                print(f"Image not found for {class_name}/{ann_file}")
                continue
            
            # 获取图片尺寸
            try:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"Error opening {img_path}: {e}")
                continue
            
            # 读取标注文件
            ann_path = os.path.join(class_ann_dir, ann_file)
            with open(ann_path, 'r') as f:
                lines = f.readlines()
            
            # 转换每个标注
            yolo_lines = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    print(f"Invalid line in {ann_path}: {line}")
                    continue
                
                # 解析坐标
                try:
                    l = int(parts[1])
                    t = int(parts[2])
                    w = int(parts[3])
                    h = int(parts[4])
                except ValueError:
                    print(f"Invalid numbers in {ann_path}: {line}")
                    continue
                
                # 计算归一化坐标
                x_center = (l + w/2) / img_width
                y_center = (t + h/2) / img_height
                norm_w = w / img_width
                norm_h = h / img_height
                
                # 边界检查
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                    print(f"Warning: Bbox out of bounds in {ann_path}")
                
                yolo_line = f"{class_mapping[class_name]} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
                yolo_lines.append(yolo_line)
            
            # 写入输出文件（保持原始文件名）
            output_path = os.path.join(class_output_dir, f"{base_name}.txt")
            with open(output_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
            # print(f"Converted {class_name}/{ann_file} -> {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', 
                        default='/home/nrc505/myyolov8/datasets/EXDark/ExDark_Annno/ExDark_Annno',
                        help='Path to annotations (default: %(default)s)')
    parser.add_argument('--img',
                        default='/home/nrc505/myyolov8/datasets/EXDark/ExDark/ExDark',
                        help='Path to images (default: %(default)s)')
    parser.add_argument('--output',
                        default='/home/nrc505/myyolov8/datasets/EXDark/labels',
                        help='Output directory (default: %(default)s)')
    args = parser.parse_args()
    
    # 打印使用的参数值
    print("\n" + "="*40)
    print(f"Using annotations from: {args.ann}")
    print(f"Using images from:      {args.img}")
    print(f"Saving output to:       {args.output}")
    print("="*40 + "\n")
    
    convert_bbgt_to_yolo(args.ann, args.img, args.output)