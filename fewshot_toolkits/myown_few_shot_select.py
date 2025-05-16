import json
import os
from collections import defaultdict
import random 
from tqdm import tqdm
def load_coco_annotations(data_root, img_dir, json_file):
    """加载COCO标注文件并解析图像路径"""
    with open(os.path.join(data_root,json_file), 'r') as f:
        annotations = json.load(f)
    
    # 构造图像路径
    # img_dir = os.path.join(data_root, img_dir)
    # for img in annotations['images']:
    #     img['file_path'] = os.path.join(img_dir, img['file_name'])  # 将完整的图像路径加到每个图像字典中

    return annotations

def build_image_id_to_annotations(annotations):
    """建立image_id到annotations的索引"""
    image_id_to_annotations = defaultdict(list)
    for ann in annotations['annotations']:
        image_id_to_annotations[ann['image_id']].append(ann)
    return image_id_to_annotations

def get_category_to_annotations(annotations):
    """将标注按类别ID分组"""
    category_to_annotations = defaultdict(list)
    for ann in annotations['annotations']:
        category_to_annotations[ann['category_id']].append(ann)
    return category_to_annotations

def select_images_by_category_shots(annotations, shot_per_category,sparse=True):
    """按类别选择足够的标注，确保每个类别有shot_per_category个标注"""
    image_id_to_annotations = build_image_id_to_annotations(annotations)
    category_to_annotations = get_category_to_annotations(annotations)
    #print(len(category_to_annotations[7]))
    # print(annotations['categories'])
    # print(type(annotations['categories'][0]['id']))
    # import sys  
    # sys.exit()
    # prefer_categories = [7,8,6,2]
    prefer_categories = [2,7,8,6,9]
    category_counts = defaultdict(int)
    selected_images = set()
    selected_annotations = []
    max_ann_per_image = 12
    # 预先选择 prefer_categories 的图像和标注
    for category_id in prefer_categories:
        prefer_annotations = category_to_annotations[category_id]
        random.shuffle(prefer_annotations)  # 加这一行
        # 计算需要选择的标注数量
        required_count = int(0.9 * shot_per_category)
        # print(category_counts[7])
        # import sys
        # sys.exit()
        # import pdb 
        # pdb.set_trace()
        # 遍历 prefer_annotations，以图像为单位选择标注
        for ann in prefer_annotations:
            image_id = ann['image_id']
            if image_id not in selected_images:
                # 将该图像的所有标注一并添加
                image_category_counts = defaultdict(int)
                annotations_for_image = image_id_to_annotations[image_id]
                for ann in annotations_for_image:
                    sub_category_id = ann['category_id']
                    image_category_counts[sub_category_id] += 1

                if sparse:
                    theimg_total_count = sum(image_category_counts.values())
                    if theimg_total_count > max_ann_per_image:
                        continue

                will_exceed = False
                for sub_category_id, count in image_category_counts.items():
                    if sub_category_id == 3 and count > 5:
                        will_exceed = True
                        break
                    if category_counts[sub_category_id] + count > shot_per_category:
                        will_exceed = True
                        break
                if will_exceed:
                    continue
                selected_annotations.extend(annotations_for_image)
                selected_images.add(image_id)
                for sub_category_id, count in image_category_counts.items():
                    category_counts[sub_category_id] += count
                # print(category_counts)
                # 如果达到 required_count，停止选择
                if category_counts[category_id]  >= required_count:
                    break

    max_rounds = 5
    diff_threshold = 10
    max_ann_per_image = 10
    # 获取所有图像的列表
    # 遍历每个图像
    for round_idx in range(max_rounds):
        images = annotations['images']
        random.shuffle(images)
        for img in images:
            image_id = img['id']
            if image_id in selected_images:
                continue
            image_category_counts = defaultdict(int)
            annotations_for_image = image_id_to_annotations[image_id]
            for ann in annotations_for_image:
                category_id = ann['category_id']
                image_category_counts[category_id] += 1
            if sparse:
                theimg_total_count = sum(image_category_counts.values())
                if theimg_total_count > max_ann_per_image + 4*round_idx :
                    continue
            will_exceed = False
            for category_id, count in image_category_counts.items():
                # if category_id == 3 and count > 5:
                #     will_exceed = True
                #     break
                # 特别处理类别3（车），限制其数量
                if category_counts[category_id] + count > shot_per_category:
                    will_exceed = True
                    break
            if will_exceed:
                continue
            selected_annotations.extend(annotations_for_image)
            selected_images.add(image_id)
            for category_id, count in image_category_counts.items():
                category_counts[category_id] += count

        # 统计当前采样状态
        selected_category_counts = defaultdict(int)
        for ann in selected_annotations:
            selected_category_counts[ann['category_id']] += 1
        diff = 0
        for cat in annotations['categories']:
            cat_id = cat['id']
            diff += max(0, shot_per_category - selected_category_counts.get(cat_id, 0))
        print(f"Round {round_idx+1}: diff={diff}")
        if diff < diff_threshold:
            break

    selected_images_info = [img for img in annotations['images'] if img['id'] in selected_images]

    return selected_images_info, selected_annotations, dict(selected_category_counts), diff

def select_images_by_number(annotations, num_images):
    """按图像数量选择图像，确保选择的图像能够覆盖所有类别"""
    category_to_annotations = get_category_to_annotations(annotations)
    image_id_to_annotations = build_image_id_to_annotations(annotations)
    selected_images = set()
    selected_annotations = set()

    all_categories = set(category_to_annotations.keys())
    selected_images_info = []
    selected_anns = []

    image_usage_count = defaultdict(int)
    images = annotations['images']
    random.shuffle(images)
    for img in images:
        img_annotations = image_id_to_annotations[img['id']]
        img_categories = set(ann['category_id'] for ann in img_annotations)

        if img_categories & all_categories:
            selected_images.add(img['id'])
            selected_images_info.append(img)
            selected_anns.extend(img_annotations)
            all_categories -= img_categories
        
        if len(selected_images) >= num_images:
            break
    # 统计selected_annotations中每个类别的数量
    selected_category_counts = defaultdict(int)
    for ann in selected_anns:
        selected_category_counts[ann['category_id']] += 1
    # 计算diff
    diff = 0
    for cat in annotations['categories']:
        cat_id = cat['id']
        diff += 0

    return selected_images_info, selected_anns, dict(selected_category_counts), diff

def save_few_shot_json(selected_images_info, selected_annotations, annotations, output_dir, x,mode="images",seed=0):
    """保存选择的图像和标注信息为json文件"""
    # 更新images和annotations部分
    images_output = [img for img in annotations['images'] if img['id'] in [i['id'] for i in selected_images_info]]
    annotations_output = [ann for ann in annotations['annotations'] if ann['image_id'] in [img['id'] for img in selected_images_info]]

    updated_json = {
        'images': images_output,
        'annotations': annotations_output,
        'categories': annotations['categories'],  # 保持categories不变
    }

    # 保存更新后的json
    if mode == "images":
        with open(os.path.join(output_dir, f'seed-{seed}-true-few-shot-{x}-image.json'), 'w') as f:
            json.dump(updated_json, f)
    else:
        with open(os.path.join(output_dir, f'seed-{seed}-true-few-shot-{x}-anno.json'), 'w') as f:
            json.dump(updated_json, f)

def select(data_root, img_dir, json_file, output_dir, x, seed=0,num_images=None):
    if seed is not None:
        random.seed(seed)
    annotations = load_coco_annotations(data_root, img_dir, json_file)
    mode = None
    selected_images_info=None
    selected_annotations=None 
    cat_counts=None 
    diff=None
    # 根据设置选择图像和标注
    if num_images:
        mode = "images"
        #对image来说，cat_counts和diff没有意义
        selected_images_info, selected_annotations, cat_counts, diff = select_images_by_number(annotations, num_images)
    else:
        mode = "cat"
        selected_images_info, selected_annotations, cat_counts, diff = select_images_by_category_shots(annotations, x)
    
    print(f"Seed {seed}: diff={diff}, cat_counts={cat_counts}",flush=True)
    # 保存更新后的图像和标注json
    save_few_shot_json(selected_images_info, selected_annotations, annotations, output_dir, x,mode,seed)
    return seed,diff

data_root = '/home/wuke_2024/datasets/original_datasets/'
img_dir = 'VisDrone2019-DET-train/'
json_file = "train.json"
output_dir = "/home/wuke_2024/datasets/original_datasets/loop_sparse_select_20shots"
os.makedirs(output_dir, exist_ok=True)
# main(data_root,img_dir,json_file,output_dir,x=20,num_images=20) #20张图
best_seed = None
min_diff = float('inf')
seeds = random.sample(range(1, 1000000), 5000)  # 196个随机种子
for seed in tqdm(seeds):
    use_seed,diff=select(data_root,img_dir,json_file,output_dir,x=20,seed=seed) #20个对象
    if diff < min_diff:
        min_diff = diff
        best_seed = seed
print(f"Best seed: {best_seed}, min diff: {min_diff}",flush=True)