import os
import random
import json
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm
import copy

# 配置
ann_path = 'annotations/instances_train2017.json'
img_dir = 'images/'
output_img_dir = img_dir  # 覆盖原图
output_ann_path = 'annotations/instances_train2017_augmented.json'
few_shot = 10  # 目标每类数量

# 加载COCO数据
coco = COCO(ann_path)

# 统计每类ann数量
cat_ann_count = {cat_id: 0 for cat_id in coco.getCatIds()}
for ann in coco.dataset['annotations']:
    cat_ann_count[ann['category_id']] += 1

# 找出不足few_shot的类别
cats_to_augment = [cat_id for cat_id, count in cat_ann_count.items() if count < few_shot]
print("需要增强的类别:", cats_to_augment)

# 复制COCO数据用于修改
new_dataset = copy.deepcopy(coco.dataset)
new_ann_id = max([ann['id'] for ann in coco.dataset['annotations']]) + 1

def get_non_overlapping_bbox(target_anns, w, h, box_w, box_h, max_try=50):
    """在目标图像上随机找一个不重叠的位置"""
    for _ in range(max_try):
        x = random.randint(0, w - box_w)
        y = random.randint(0, h - box_h)
        bbox = [x, y, box_w, box_h]
        overlap = False
        for ann in target_anns:
            tx, ty, tw, th = ann['bbox']
            if not (x+box_w < tx or x > tx+tw or y+box_h < ty or y > ty+th):
                overlap = True
                break
        if not overlap:
            return bbox
    return None  # 没找到合适位置

for cat_id in cats_to_augment:
    # 获取该类别所有实例
    ann_ids = coco.getAnnIds(catIds=[cat_id])
    anns = coco.loadAnns(ann_ids)
    if not anns:
        continue
    needed = few_shot - len(anns)
    print(f"类别{cat_id} 需要补充{needed}个样本")
    for i in range(needed):
        # 随机选择一个实例和对应图片
        ann = random.choice(anns)
        img = coco.loadImgs([ann['image_id']])[0]
        img_path = os.path.join(img_dir, img['file_name'])
        image = Image.open(img_path).convert('RGBA')

        # 裁剪实例
        x, y, w, h = map(int, ann['bbox'])
        instance_crop = image.crop((x, y, x + w, y + h))

        # 随机选择一张目标图片（覆盖原图）
        target_img_info = random.choice(new_dataset['images'])
        target_img_path = os.path.join(img_dir, target_img_info['file_name'])
        target_image = Image.open(target_img_path).convert('RGBA')

        # 获取目标图片已有的anns
        target_ann_ids = [a['id'] for a in new_dataset['annotations'] if a['image_id'] == target_img_info['id']]
        target_anns = [a for a in new_dataset['annotations'] if a['id'] in target_ann_ids]

        # 寻找合适位置
        pos_bbox = get_non_overlapping_bbox(target_anns, target_image.width, target_image.height, w, h)
        if pos_bbox is None:
            print("未找到合适位置，跳过")
            continue

        # 粘贴（覆盖原图）
        target_image.paste(instance_crop, (pos_bbox[0], pos_bbox[1]), instance_crop)
        target_image = target_image.convert('RGB')  # 转回RGB，避免透明通道

        # 保存覆盖后的图片
        target_image.save(target_img_path)

        # 添加新ann
        new_ann = copy.deepcopy(ann)
        new_ann['id'] = new_ann_id
        new_ann['image_id'] = target_img_info['id']
        new_ann['bbox'] = [pos_bbox[0], pos_bbox[1], w, h]
        new_dataset['annotations'].append(new_ann)

        new_ann_id += 1

# 保存新的json
with open(output_ann_path, 'w') as f:
    json.dump(new_dataset, f)

print("增强完成！")