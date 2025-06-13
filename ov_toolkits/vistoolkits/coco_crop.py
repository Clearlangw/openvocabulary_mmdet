import json
import os
from pycocotools.coco import COCO
from PIL import Image
import shutil # 用于清空文件夹

def crop_coco_objects(data_dir, ann_file_name, image_folder_name,output_dir, clear_output_dir=False):
    """
    裁剪COCO数据集中标注的对象，并按类别保存到不同文件夹。

    参数:
    data_dir (str): 包含图像文件夹和注释文件的根目录。
                     例如: 'coco_dataset/'
                     该目录下应有 'images/' (或类似的图像文件夹名) 和 ann_file_name
    ann_file_name (str): COCO JSON注释文件的名称。
                         例如: 'annotations/instances_train2017.json'
    output_dir (str): 保存裁剪后目标图像的输出根目录。
                      例如: 'cropped_objects/'
    clear_output_dir (bool): 是否在开始前清空输出目录。默认为 False。
    """

    # 构造注释文件的完整路径
    ann_file_path = os.path.join(data_dir, ann_file_name)

    if not os.path.isdir(os.path.join(data_dir, image_folder_name)):
        # 如果推断的文件夹不存在，尝试常见的 'images' 文件夹
        if os.path.isdir(os.path.join(data_dir, 'images')):
            image_folder_name = 'images'
        else:
            # 如果都没有，提示用户手动指定或检查结构
            print(f"警告: 无法自动确定图像文件夹。请确保图像在 '{data_dir}/{image_folder_name}' 或 '{data_dir}/images' 中，或者修改脚本以指定正确的图像路径。")
            # 你也可以在这里引发一个错误: raise ValueError("无法找到图像文件夹")
            # 或者，让用户传入 image_dir 参数
            # image_dir = os.path.join(data_dir, 'your_image_folder_name') # 示例
            # 为了脚本能继续运行（尽管可能找不到图片），我们暂时不中断
            import sys
            sys.exit()
            pass # 允许继续，但可能会在加载图像时出错

    image_dir = os.path.join(data_dir, image_folder_name)
    print(f"图像文件夹路径设置为: {image_dir}")


    # 如果输出目录已存在且需要清空
    if clear_output_dir and os.path.exists(output_dir):
        print(f"正在清空输出目录: {output_dir}")
        shutil.rmtree(output_dir) # 删除整个目录树
    
    # 创建输出根目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出根目录: {output_dir}")

    # 初始化COCO API
    try:
        coco = COCO(ann_file_path)
    except Exception as e:
        print(f"加载COCO注释文件失败: {e}")
        print(f"请确保路径 '{ann_file_path}' 正确，并且文件是有效的COCO JSON格式。")
        return

    # 获取所有类别信息
    cats = coco.loadCats(coco.getCatIds())
    cat_names = {cat['id']: cat['name'] for cat in cats}
    cat_names_reverse = {cat['name']: cat['id'] for cat in cats} # 用于创建文件夹

    # 为每个类别创建输出子文件夹
    for cat_name in cat_names_reverse.keys():
        cat_output_dir = os.path.join(output_dir, cat_name)
        if not os.path.exists(cat_output_dir):
            os.makedirs(cat_output_dir)
            print(f"已创建类别文件夹: {cat_output_dir}")

    # 获取所有图像ID
    img_ids = coco.getImgIds()
    print(f"共找到 {len(img_ids)} 张图像的标注信息。")

    processed_objects_count = 0
    # 遍历所有图像
    for i, img_id in enumerate(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_file_name = img_info['file_name']
        img_path = os.path.join(image_dir, img_file_name)

        if i % 100 == 0 and i > 0: # 每处理100张图片打印一次进度
            print(f"正在处理第 {i+1}/{len(img_ids)} 张图像: {img_file_name}")

        # 检查图像文件是否存在
        if not os.path.exists(img_path):
            print(f"警告: 图像文件未找到 {img_path}。跳过此图像。")
            # 尝试在其他常见位置查找，例如直接在 data_dir 下
            common_img_path = os.path.join(data_dir, img_file_name)
            if os.path.exists(common_img_path):
                img_path = common_img_path
                print(f"信息: 在备用路径找到了图像: {img_path}")
            else:
                # 尝试去掉路径中的子目录，如 'train2017/image.jpg' -> 'image.jpg'
                # COCO file_name 有时会包含子目录
                base_img_file_name = os.path.basename(img_file_name)
                search_paths = [
                    os.path.join(data_dir, base_img_file_name), # data_dir/image.jpg
                    os.path.join(data_dir, image_folder_name, base_img_file_name) # data_dir/train2017/image.jpg
                ]
                found_img = False
                for p in search_paths:
                    if os.path.exists(p):
                        img_path = p
                        print(f"信息: 在备用路径 '{p}' 找到了图像。")
                        found_img = True
                        break
                if not found_img:
                    print(f"警告: 在多个路径尝试后仍未找到图像文件 {img_file_name}。跳过此图像。")
                    continue


        try:
            # 打开图像
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"错误: 图像文件 {img_path} 未找到。跳过。")
            continue
        except Exception as e:
            print(f"错误: 打开图像 {img_path} 失败: {e}。跳过。")
            continue

        # 获取该图像的所有标注ID
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # 遍历该图像的所有标注
        for ann_idx, ann in enumerate(anns):
            category_id = ann['category_id']
            category_name = cat_names.get(category_id, 'unknown_category') # 获取类别名称，如果ID未知则标记为unknown

            # 获取边界框 [x, y, width, height]
            bbox = ann['bbox']
            x, y, w, h = [int(coord) for coord in bbox] # 确保坐标是整数

            # 检查边界框的有效性
            if w <= 0 or h <= 0:
                print(f"警告: 图像 {img_file_name} 中的一个标注具有无效的边界框 (w={w}, h={h})。跳过此标注。")
                continue

            # 裁剪目标
            # PIL的crop方法参数是 (left, upper, right, lower)
            # COCO的bbox是 [x, y, width, height]
            # x_min, y_min, width, height
            # 所以 right = x + width, lower = y + height
            cropped_image = image.crop((x, y, x + w, y + h))

            # 构建保存路径和文件名
            # 文件名可以包含图像ID、标注ID和类别，以确保唯一性
            output_filename = f"{os.path.splitext(os.path.basename(img_file_name))[0]}_obj{ann['id']}_{category_name}.png"
            output_cat_dir = os.path.join(output_dir, category_name)
            # 再次确保类别目录存在 (以防 'unknown_category' 的情况)
            if not os.path.exists(output_cat_dir):
                os.makedirs(output_cat_dir)

            save_path = os.path.join(output_cat_dir, output_filename)

            try:
                # 保存裁剪后的图像
                cropped_image.save(save_path)
                processed_objects_count += 1
            except Exception as e:
                print(f"错误: 保存裁剪图像 {save_path} 失败: {e}")

    print(f"\n处理完成！共裁剪并保存了 {processed_objects_count} 个目标。")
    print(f"裁剪后的目标图像保存在: {output_dir}")

# --- 使用示例 ---
if __name__ == '__main__':
    # 请根据你的实际路径修改这些变量
    # COCO数据集的根目录，例如 'D:/datasets/coco2017'
    # 这个目录下应该有一个包含图像的文件夹 (如 'train2017', 'val2017')
    # 以及一个包含注释文件的文件夹 (如 'annotations')
    coco_data_directory = '/home/wuke_2024/datasets/original_datasets/' 
    image_folder_name = 'VisDrone2019-DET-train'
    # COCO注释文件的名称 (相对于 coco_data_directory)
    # 例如 'annotations/instances_train2017.json' 或 'annotations/instances_val2017.json'
    annotation_file = 'train.json' 
    
    # 输出裁剪后图像的根目录
    output_directory = '/home/wuke_2024/datasets/VisDrone2019-Cropped'

    # 是否在开始前清空输出目录
    clear_existing_output = True 

    # 检查用户是否修改了路径 (简单的占位符检查)
    if 'path/to/your' in coco_data_directory or 'path/to/your' in output_directory or 'your_coco_annotations.json' in annotation_file:
        print("错误: 请在脚本中修改 'coco_data_directory', 'annotation_file', 和 'output_directory' 为你的实际路径！")
        print("示例:")
        print("coco_data_directory = 'D:/COCO_Dataset'")
        print("annotation_file = 'annotations/instances_val2017.json'")
        print("output_directory = './coco_cropped_objects'")
    else:
        print("开始处理...")
        crop_coco_objects(
            data_dir=coco_data_directory,
            ann_file_name=annotation_file,
            image_folder_name=image_folder_name,
            output_dir=output_directory,
            clear_output_dir=clear_existing_output
        )
