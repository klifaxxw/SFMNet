_base_ = [
    "../../_base_/datasets/loveda.py",
    "../../_base_/default_runtime.py",
    "../../_base_/schedules/schedule_160k_adamw.py",
]
# model settings
total_iters = 48100
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='VMUBackbone',
        pretrained_ckpt="weights/vm_unet_tiny_e292.pth",
        init_cfg=None
        ),
    decode_head=dict(
        type='VMUHead',
        num_classes=6,
        pretrained_ckpt="weights/vm_unet_tiny_e292.pth",
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

data = dict(samples_per_gpu=6)
evaluation = dict(interval=3000, metric='mIoU')
checkpoint_config = dict(by_epoch=False, interval=6000)
runner = dict(type='IterBasedRunner', max_iters=total_iters)
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'norm_layer': dict(decay_mult=0.),
                                                'norm': dict(decay_mult=0.),
                                                'A_log':dict(decay_mult=0.0),  # 禁用 A_log 的权重衰减
                                                'D': dict(decay_mult=0.0),  # 禁用 D 的权重衰减
                                                'absolute_pos_embed': dict(decay_mult=0.0),
                                                'relative_position_bias_table': dict(decay_mult=0.0),
                                                }))
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
find_unused_parameters = True