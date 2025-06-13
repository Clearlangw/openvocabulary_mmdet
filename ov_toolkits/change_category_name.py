import json
import os
import argparse

def modify_json_categories(json_file_path):
    """
    修改JSON文件中的categories字段，并另存为新文件
    
    Args:
        json_file_path: JSON文件路径
    """
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 指定的新categories值
    new_categories = [
        {"id": 0, "name": "pedestrian XX XX", "supercategory": "person"},
        {"id": 1, "name": "people XX XX", "supercategory": "person"},
        {"id": 2, "name": "bicycle XX XX", "supercategory": "bicycle"},
        {"id": 3, "name": "car XX XX", "supercategory": "car"},
        {"id": 4, "name": "van XX XX", "supercategory": "truck"},
        {"id": 5, "name": "truck XX XX", "supercategory": "truck"},
        {"id": 6, "name": "tricycle XX", "supercategory": "motor"},
        {"id": 7, "name": "XX XX XX", "supercategory": "motor"},
        {"id": 8, "name": "bus XX XX", "supercategory": "bus"},
        {"id": 9, "name": "motor XX XX", "supercategory": "motor"}
    ]
    
    # 替换categories字段
    data['categories'] = new_categories
    
    # 生成新文件名
    file_dir = os.path.dirname(json_file_path)
    file_name = os.path.basename(json_file_path)
    new_file_name = os.path.join(file_dir, "shine_diff_" + file_name)
    
    # 保存修改后的JSON文件
    with open(new_file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    
    print(f"已将修改后的文件保存为: {new_file_name}")

def main():
    train_json_file = "/home/wuke_2024/datasets/original_datasets/train.json"
    val_json_file = "/home/wuke_2024/datasets/original_datasets/val.json"  
    
    modify_json_categories(train_json_file)
    modify_json_categories(val_json_file)

if __name__ == "__main__":
    main() 