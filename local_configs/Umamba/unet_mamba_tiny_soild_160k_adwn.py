_base_ = [
    # '../_base_/models/upernet_mamba.py',
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
        type='MambaUE',
        patch_size=4,
        in_chans=3,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        d_state=16,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        patch_norm=True,
        use_checkpoint=False,
        out_indices=(0, 1, 2, 3),
        pretrained=None,
        ape=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/dell/x/model/bdfomer/weights/vmamba_tiny_e292.pth'),
        ),
    decode_head=dict(
        type='MambaUD',
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        d_state=16,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        use_checkpoint=False,
        dropout_ratio=0.1,
        num_classes=2,
        align_corners=False,
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        channels=256,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

data = dict(samples_per_gpu=6)
evaluation = dict(interval=8000, metric='mIoU')
checkpoint_config = dict(by_epoch=False, interval=8000)
runner = dict(type='IterBasedRunner', max_iters=170000)
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.),
                                                'A_log':dict(decay_mult=0.0),  # 禁用 A_log 的权重衰减
                                                'D': dict(decay_mult=0.0),  # 禁用 D 的权重衰减
                                                 }))
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
find_unused_parameters = True