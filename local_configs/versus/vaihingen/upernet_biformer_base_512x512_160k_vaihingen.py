
_base_ = [
    "../../_base_/datasets/vaihingen.py",
    '../../_base_/default_runtime.py',
    "../../_base_/schedules/schedule_160k_adamw.py",
]

total_iters = 48100
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='BiFormer',
        depth=[4, 4, 18, 4],
        embed_dim=[96, 192, 384, 768],
        head_dim=32,
        qk_dims=[96, 192, 384, 768],
        drop_path_rate=0.4,
        n_win=7,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        param_routing=False,
        diff_routing=False,
        soft_routing=False,
        pre_norm=True,
        pe=None,
        pe_stages=[0],
        auto_pad=True,
        kv_downsample_kernels=[4, 2, 1, 1],
        kv_downsample_ratios=[4, 2, 1, 1],
        mlp_ratios=[3, 3, 3, 3],
        param_attention='qkvo',
        mlp_dwconv=False,
        norm_cfg=norm_cfg,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint='weights/biformer_base_best.pth'),
    ),
     decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                    'relative_position_bias_table': dict(decay_mult=0.),
                                    'norm': dict(decay_mult=0.)}))

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

data = dict(samples_per_gpu=4)
evaluation = dict(interval=3000, metric='mIoU')
checkpoint_config = dict(by_epoch=False, interval=9000)
runner = dict(type='IterBasedRunner', max_iters=total_iters)
