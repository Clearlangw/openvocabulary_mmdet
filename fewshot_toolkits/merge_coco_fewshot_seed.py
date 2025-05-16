from dectron_generate_few_shot_splits import modify_dataset_for_few_shot
import os
# train_json_path = '/home/wuke_2024/datasets/cocosplit/datasplit/trainvalno5k.json'
train_json_path = os.path.join('/home/wuke_2024/datasets','cocosplit','datasplit','trainvalno5k.json')
base_split_path = os.path.join('/home/wuke_2024/datasets','cocosplit')
shots = [1,2,3,5,10,30]
seed = 0
for shot in shots:
    modify_dataset_for_few_shot(train_json_path,base_split_path,shot,seed)