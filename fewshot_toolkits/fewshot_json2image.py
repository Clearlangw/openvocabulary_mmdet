import os
import json
import shutil
from tqdm import tqdm
# 你的变量
image_root = '/home/wuke_2024/datasets/original_datasets/'
train_image_dir = "VisDrone2019-DET-train"
val_image_dir = "VisDrone2019-DET-val"
json_list = ['/home/wuke_2024/datasets/original_datasets/myown_select_20shots_json/seed-517600-true-few-shot-20-anno.json']



def process_json2image(image_root, train_image_dir, val_image_dir, json_path,all_from_train=True,has_images_sub_dir=True):
    # 获取json文件名（不带后缀）
    json_name = os.path.splitext(os.path.basename(json_path))[0]
    sub_dir = 'images' if has_images_sub_dir else ''
    # 读取json
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 遍历images
    for img in tqdm(data['images']):
        file_name = img['file_name']
        if not all_from_train:
            # 判断是train还是val
            if 'train' in file_name:
                src_dir = os.path.join(image_root, train_image_dir)
                target_dir = os.path.join(image_root, json_name)
            elif 'val' in file_name:
                src_dir = os.path.join(image_root, val_image_dir)
                target_dir = os.path.join(image_root, json_name)
            else:
                print(f"未识别的file_name: {file_name}")
                continue
        else:
            src_dir = os.path.join(image_root, train_image_dir)
            target_dir = os.path.join(image_root, json_name)
        # 创建目标文件夹
        os.makedirs(target_dir, exist_ok=True)
        if has_images_sub_dir:
            os.makedirs(os.path.join(target_dir,sub_dir),exist_ok=True)
        # 源文件路径
        src_path = os.path.join(src_dir, file_name)
        # 目标文件路径
        dst_path = os.path.join(target_dir, file_name)

        # 检查源文件是否存在
        if not os.path.exists(src_path):
            print(f"文件不存在: {src_path}")
            continue

        # 复制文件
        shutil.copy2(src_path, dst_path)
        print(f"已复制: {src_path} -> {dst_path}")

    new_json_path = os.path.join(image_root,os.path.basename(json_path))
    shutil.copy2(json_path, new_json_path)
    print("处理完成。")



for json_path in json_list:
    print(json_path)
    process_json2image(image_root, train_image_dir, val_image_dir, json_path)
