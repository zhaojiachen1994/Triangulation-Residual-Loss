_base_ = [
    '../../default_runtime.py',
    '../../datasets/human_p17.py'
]


load_from = "work_dirs/h36m/CDTriangNet_score_hrnet_w48_h36m/best_MPJPE_epoch_4.pth"
total_epochs = 60

evaluation = dict(interval=2, metric='mpjpe', by_epoch=True, save_best='MPJPE')
optimizer = dict(type='Adam', lr=1e-5, )
optimizer_config = dict(grad_clip=None)

lr_config = dict(policy='step', warmup='linear', warmup_iters=50, warmup_ratio=0.001, step=[17, 20])

log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook'),])
channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
    space_size=[4000, 4000, 3000],
    space_center=[0, 0, 0],
    cube_size=[40, 40, 30],
    num_cameras=4,
    use_different_joint_weights=False)

num_joints = 17
model = dict(
    type='TriangNet',
    pretrained=None,
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384))),
    ),

    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=48,
        out_channels=num_joints,
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),

    score_head=dict(
        type='GlobalAveragePoolingHead',
        in_channels=48,
        n_classes=num_joints
    ),

    triangulate_head=dict(
        type='TriangulateHead',
        num_cams=4,
        img_shape=[256, 256],
        heatmap_shape=[64, 64],
        softmax_heatmap=True,
        loss_3d_sup=dict(type='MSELoss',
                         use_target_weight=False,
                         loss_weight=1.),
        det_conf_thr=0.0,
    ),
    train_cfg=dict(
        use_2d_sup=True,  # use the 2d ground truth to train keypoint_head
        use_3d_sup=False,  # use the 3d ground truth to train triangulate_head
        use_3d_unsup=True,  # use the triangulation residual loss to train triangulate_head
        score_thr=[0.3, 0.7]
    ),
    test_cfg=dict(
        flip_test=False,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11)
)

train_pipeline = [
    dict(
        type="MultiItemProcess",
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='SquareBbox'),
            dict(type='CropImage',
                 update_camera=True),
            dict(type='ResizeImage',
                 update_camera=True),
            dict(type='ComputeProjMatric'),
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(type='TopDownGenerateTarget', sigma=2)
        ]),
    dict(
        type='DiscardDuplicatedItems',
        keys_list=['dataset', 'ann_info', 'joints_4d', 'joints_4d_visible']
    ),
    dict(
        type='GroupCams',
        keys=['img', 'target', 'target_weight', 'proj_mat', 'joints_3d', 'bbox']
    ),
    dict(
        type="Collect",
        keys=['img', 'target', 'target_weight', 'joints_4d', 'proj_mat', 'joints_3d', 'joints_4d_visible'],
        meta_keys=['image_file', 'bbox_offset', 'resize_ratio']
    )
]
val_pipeline = [
    dict(
        type="MultiItemProcess",
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='SquareBbox'),
            dict(type='CropImage',
                 update_camera=True),
            dict(type='ResizeImage',
                 update_camera=True),
            dict(type='ComputeProjMatric'),
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]),
    dict(
        type='DiscardDuplicatedItems',
        keys_list=['dataset', 'ann_info', 'joints_4d', 'joints_4d_visible']
    ),
    dict(
        type='GroupCams',
        keys=['img', 'proj_mat']
    ),
    dict(
        type="Collect",
        keys=['img', 'proj_mat'],
        meta_keys=['image_file', 'bbox_offset', 'resize_ratio',
                   'joints_4d', 'joints_4d_visible']
    )
]
test_pipeline = val_pipeline

h36m_root = "D:/Datasets/h36m_dataset/human3.6m_parse"


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=4),
    test_dataloader=dict(samples_per_gpu=4),
    train=dict(
            type='TopDownH36MDataset',
            ann_file=f'{h36m_root}/annotations_old3/train/Human36M_train_joint2d_1hz.json',
            img_prefix=f'{h36m_root}/images/',
            data_cfg=data_cfg,
            pipeline=train_pipeline,
            dataset_info={{_base_.dataset_info}}),
    val=dict(
            type="Body3DH36MMviewDataset",
            ann_file=f"{h36m_root}/annotations_old3/test/Human36M_test_joint2d_0.1hz.json",
            ann_3d_file=f"{h36m_root}/annotations_old3/test/Human36M_test_joint3d_0.1hz.json",
            cam_file=f"{h36m_root}/annotations/cameras.json",
            img_prefix=f"{h36m_root}/images/",
            data_cfg=data_cfg,
            pipeline=val_pipeline,
            dataset_info={{_base_.dataset_info}}),
    test=dict(
            type="Body3DH36MMviewDataset",
            ann_file=f"{h36m_root}/annotations_old3/test/Human36M_test_joint2d_10hz.json",
            ann_3d_file=f"{h36m_root}/annotations_old3/test/Human36M_test_joint3d_10hz.json",
            cam_file=f"{h36m_root}/annotations/cameras.json",
            img_prefix=f"{h36m_root}/images/",
            data_cfg=data_cfg,
            pipeline=test_pipeline,
            dataset_info={{_base_.dataset_info}}),
)