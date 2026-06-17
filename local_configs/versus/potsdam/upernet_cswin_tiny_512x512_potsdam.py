_base_ = [
    '../_base_/models/upernet_cswin.py', "../_base_/datasets/potsdam.py",
    '../_base_/default_runtime.py', "../_base_/schedules/schedule_160k_adamw.py",
]
total_iters = 32000
model = dict(
    backbone=dict(
        type='CSWin',
        embed_dim=64,
        depth=[1,2,21,1],
        num_heads=[2,4,8,16],
        split_size=[1,2,7,7],
        drop_path_rate=0.3,
        use_chk=False,
    ),
    decode_head=dict(
        in_channels=[64,128,256,512],
        num_classes=6
    ),
    auxiliary_head=dict(
        in_channels=256,
        num_classes=6
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

data = dict(samples_per_gpu=12)
evaluation = dict(interval=3000, metric='mIoU')
checkpoint_config = dict(by_epoch=False, interval=3000)
runner = dict(type='IterBasedRunner', max_iters=total_iters)
