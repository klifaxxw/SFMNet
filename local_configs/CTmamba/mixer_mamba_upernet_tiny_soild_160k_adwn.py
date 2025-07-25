_base_ = [
    "../_base_/datasets/soild.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_160k_adamw.py",
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='mixer',
        in_chans=3,
        depths=[2, 2, 9, 4],
        dims=[64, 144, 288, 512],
        num_heads=[2, 4, 8, 16],
        mlp_ratios=[4, 4, 4, 4],
        kernels=[7, 11, 21],
        num_points=[4, 6, 8, 12],
        drop_rate=0.,
        drop_path_rate=0.1,
        patch_norm=norm_cfg,
        pos_func="relative",
        use_checkpoint=False,
        ssm_ratio=1,
        scale=False,
        out_indices=(0, 1, 2, 3),
        pretrained=None,
        ape=False,
        init_cfg=None
        ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[64, 144, 288, 512],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=288,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

data = dict(samples_per_gpu=8)
evaluation = dict(interval=3000, metric='mIoU')
checkpoint_config = dict(by_epoch=False, interval=3000)
runner = dict(type='IterBasedRunner', max_iters=40000)
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=9e-5, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.),
                                                'A_log':dict(decay_mult=0.0),  # 禁用 A_log 的权重衰减
                                                'D': dict(decay_mult=0.0),  # 禁用 D 的权重衰减
                                                 }))
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=2000,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
find_unused_parameters = True