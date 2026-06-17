_base_ = [
    "../../_base_/datasets/potsdam.py",
    "../../_base_/default_runtime.py",
    "../../_base_/schedules/schedule_160k_adamw.py",
]
# model settings
total_iters = 48100
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='sff',
        pretrained_ckpt="weights/convnext_tiny.in12k_ft_in1k_384.pth",
        init_cfg=None
        ),
    decode_head=dict(
        type='SFFHead',
        in_channels=[96, 192, 384, 768],
        num_classes=6,
        loss_decode=dict(
            type='SFFLoss')),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

data = dict(samples_per_gpu=8)
evaluation = dict(interval=3000, metric='mIoU')
checkpoint_config = dict(by_epoch=False, interval=9000)
runner = dict(type='IterBasedRunner', max_iters=total_iters)
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01)
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
find_unused_parameters = True