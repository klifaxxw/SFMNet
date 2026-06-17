# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).

_base_ = [
    '../../_base_/models/ResNet_MSCA_EPFO_r50-d8.py',
    "../../_base_/datasets/vaihingen.py",
    "../../_base_/default_runtime.py",
    "../../_base_/schedules/schedule_160k_adamw.py"
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='/home/dell/x/model/bdfomer/weights/resnet101_v1c-e67eebb6.pth',
    backbone=dict(depth=101),
    decode_head=[
        dict(
            type='MSCA_Head',
            in_channels=2048,
            in_index=3,
            channels=512,
            pool_scales=(1, 2, 3, 6),
            use_CRF=False,
            dropout_ratio=0.1,
            num_classes=6,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='EPFO_Head',
            in_channels=[256],
            in_index=[0],
            channels=256,
            num_fcs=3,
            theta=5,
            coarse_pred_each_layer=True,
            dropout_ratio=-1,
            num_classes=6,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ],
    train_cfg=dict(ratio=0.75),
    test_cfg=dict(mode='whole', ratio=0.75))

data = dict(samples_per_gpu=8)
evaluation = dict(interval=3000, metric='mIoU')
checkpoint_config = dict(by_epoch=False, interval=9000)
runner = dict(type='IterBasedRunner', max_iters=36100)
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)