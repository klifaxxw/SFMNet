_base_ = [
    '../../_base_/datasets/vaihingen.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py',
]

# 多卡训练建议 SyncBN
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='UNetFormerBackbone',
        backbone_name='swsl_resnet18',   
        pretrained=False,
        pretrained_ckpt='/home/dell/x/model/bdfomer/weights/semi_weakly_supervised_resnet18-118f1556.pth',                
        output_stride=32,
        out_indices=(1, 2, 3, 4),
        decode_channels=64,
        window_size=8,
        norm_cfg=norm_cfg,
        convert_syncbn=True,            
    ),

    decode_head=dict(
        type='UNetFormerMainHead',
        in_channels=64,         
        channels=64,            
        num_classes=6,
        in_index=0,
        decode_channels=64,
        dropout=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),

    auxiliary_head=dict(
        type='UNetFormerAuxHead',
        in_channels=64,         
        channels=64,
        num_classes=6,
        in_index=1,
        dropout=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
    ),

    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=6e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01
)

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)

data = dict(samples_per_gpu=8)
runner = dict(type='IterBasedRunner', max_iters=36100)
checkpoint_config = dict(by_epoch=False, interval=9000)
evaluation = dict(interval=3000, metric='mIoU')