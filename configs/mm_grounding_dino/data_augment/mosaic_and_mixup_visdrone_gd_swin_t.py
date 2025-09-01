_base_ = '../grounding_dino_swin-t_pretrain_obj365.py'

data_root = '/home/wuke_2024/datasets/original_datasets/'
class_name = ("pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor")
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

model = dict(bbox_head=dict(num_classes=num_classes))

img_scale = (1333, 800)

train_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

##YOLOX那里mosaic的写法：
# train_pipeline = [
#     dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
#     dict(
#         type='RandomAffine',
#         scaling_ratio_range=(0.1, 2),
#         # img_scale is (width, height)
#         border=(-img_scale[0] // 2, -img_scale[1] // 2)),
#     dict(
#         type='MixUp',
#         img_scale=img_scale,
#         ratio_range=(0.8, 1.6),
#         pad_val=114.0),
#     dict(type='YOLOXHSVRandomAug'),
#     dict(type='RandomFlip', prob=0.5),
#     # According to the official implementation, multi-scale
#     # training is not considered here but in the
#     # 'mmdet/models/detectors/yolox.py'.
#     # Resize and Pad are for the last 15 epochs when Mosaic,
#     # RandomAffine, and MixUp are closed by YOLOXModeSwitchHook.
#     dict(type='Resize', scale=img_scale, keep_ratio=True),
#     dict(
#         type='Pad',
#         pad_to_square=True,
#         # If the image is three-channel, the pad value needs
#         # to be set separately for each channel.
#         pad_val=dict(img=(114.0, 114.0, 114.0))),
#     dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
#     dict(type='PackDetInputs')
# ]

# train_dataset = dict(
#     # use MultiImageMixDataset wrapper to support mosaic and mixup
#     type='MultiImageMixDataset',
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='annotations/instances_train2017.json',
#         data_prefix=dict(img='train2017/'),
#         pipeline=[
#             dict(type='LoadImageFromFile', backend_args=backend_args),
#             dict(type='LoadAnnotations', with_bbox=True)
#         ],
#         filter_cfg=dict(filter_empty_gt=False, min_size=32),
#         backend_args=backend_args),
#     pipeline=train_pipeline)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    # _delete_=True,
    dataset = dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='ConcatDataset',
            datasets=[dict(
                        _delete_ = True, # 删除不必要的设置
                        type='CocoDataset',
                        data_root=data_root,
                        metainfo=metainfo,
                        return_classes=True,
                        pipeline=[
                            dict(type='LoadImageFromFile'),
                            dict(type='LoadAnnotations', with_bbox=True)
                        ],
                        filter_cfg=dict(filter_empty_gt=False, min_size=32),
                        ann_file='train.json',
                        data_prefix=dict(img='VisDrone2019-DET-train/'),
                    ),]),
        pipeline=train_pipeline,
    ),
    )

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img='VisDrone2019-DET-val/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'val.json')
test_evaluator = val_evaluator

max_epoch = 50

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='coco/bbox_mAP'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[45],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.0),
            'language_model': dict(lr_mult=0.0)
        }))
load_from = '/home/wuke_2024/ov202503/mmdetection/grounding_dino_swin-t_pretrain_obj365_goldg_v3det_20231218_095741-e316e297.pth'
# load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'  # noqa
