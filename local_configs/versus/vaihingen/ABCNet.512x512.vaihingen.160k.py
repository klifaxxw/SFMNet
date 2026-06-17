_base_ = [
    '../../_base_/datasets/vaihingen.py',
    "../../_base_/default_runtime.py",
    "../../_base_/schedules/schedule_160k_adamw.py"
]
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='ABCNetBackbone',
        out_indices=(0, 1, 2),
        resnet_ckpt='weights/resnet18-5c106cde.pth',
        norm_cfg=norm_cfg,
    ),

    # 主输出：Output(256,256,n_classes, up_factor=8)
    decode_head=dict(
        type='ABCOutputHead',
        in_channels=256,
        channels=256,          
        mid_channels=256,
        num_classes=6,
        in_index=0,
        up_factor=8,
        norm_cfg=norm_cfg,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        align_corners=False,
    ),

    # 两个深监督：Output(128,64,n_classes, up_factor=8) 和 Output(128,64,n_classes, up_factor=16)
    auxiliary_head=[
        dict(
            type='ABCOutputHead',
            in_channels=128,
            channels=64,
            mid_channels=64,
            num_classes=6,
            in_index=1,
            up_factor=8,
            norm_cfg=norm_cfg,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
            align_corners=False,
        ),
        dict(
            type='ABCOutputHead',
            in_channels=128,
            channels=64,
            mid_channels=64,
            num_classes=6,
            in_index=2,
            up_factor=16,
            norm_cfg=norm_cfg,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
            align_corners=False,
        ),
    ],

    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

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

data = dict(samples_per_gpu=8)
runner = dict(type='IterBasedRunner', max_iters=36100)
checkpoint_config = dict(by_epoch=False, interval=9000)
evaluation = dict(interval=3000, metric='mIoU')
