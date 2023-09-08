dataset_info = dict(
    dataset_name='mouse_p12',
    paper_info=dict(
        author='Dunn',
        title='DANNCE',
        container='Nature Methods',
        year='2021',
        homepage='',
    ),
    keypoint_info={
        0:

            dict(
                name='left_ear_tip', id=0, color=[92, 94, 170], type='upper', swap='right_ear_tip'),
        1:
            dict(
                name='right_ear_tip',
                id=1,
                color=[92, 94, 170],
                type='upper',
                swap='left_ear_tip'),
        2:
            dict(
                name='nose',
                id=2,
                color=[92, 94, 170],
                type='upper',
                swap=''),
        3:
            dict(
                name='neck',
                id=3,
                color=[221, 94, 86],
                type='upper',
                swap=''),
        4:
            dict(name='body_middle', id=4, color=[221, 94, 86], type='upper', swap=''),
        5:
            dict(name='tail_root', id=5, color=[221, 94, 86], type='upper', swap=''),
        6:
            dict(
                name='tail_middle', id=6, color=[221, 94, 86], type='lower',
                swap=''),
        7:
            dict(
                name='tail_end', id=7, color=[221, 94, 86], type='upper', swap=''),
        8:
            dict(
                name='left_paw_end',
                id=8,
                color=[187, 97, 166],
                type='upper',
                swap='right_paw_end'),
        9:
            dict(
                name='right_paw_end',
                id=9,
                color=[109, 192, 91],
                type='upper',
                swap='left_paw_end'),
        10:
            dict(
                name='left_foot',
                id=10,
                color=[210, 220, 88],
                type='upper',
                swap='right_foot'),
        11:
            dict(
                name='right_foot',
                id=11,
                color=[98, 201, 211],
                type='lower',
                swap='left_foot'),
    },
    skeleton_info={
        0: dict(link=('tail_root', 'tail_middle'), id=0, color=[221, 94, 86]),
        1: dict(link=('tail_middle', 'tail_end'), id=1, color=[221, 94, 86]),
        2: dict(link=('tail_root', 'left_foot'), id=2, color=[210, 220, 88]),
        # 3: dict(link=('body_middle', 'left_foot'), id=3, color=[210, 220, 88]),
        3: dict(link=('tail_root', 'right_foot'), id=4, color=[98, 201, 211]),
        # 5: dict(link=('body_middle', 'right_foot'), id=5, color=[98, 201, 211]),
        4: dict(link=('tail_root', 'body_middle'), id=6, color=[221, 94, 86]),
        5: dict(link=('body_middle', 'neck'), id=7, color=[221, 94, 86]),
        6: dict(link=('neck', 'nose'), id=8, color=[221, 94, 86]),
        7: dict(link=('nose', 'right_ear_tip'), id=9, color=[92, 94, 170]),
        8: dict(link=('nose', 'left_ear_tip'), id=10, color=[92, 94, 170]),
        9: dict(link=('neck', 'left_paw_end'), id=11, color=[187, 97, 166]),
        # 12: dict(link=('body_middle', 'left_paw_end'), id=12, color=[187, 97, 166]),
        10: dict(link=('neck', 'right_paw_end'), id=13, color=[109, 192, 91]),
        # 14: dict(link=('body_middle', 'right_paw_end'), id=14, color=[109, 192, 91]),
        },
    # skeleton_info={
    #     0: dict(link=('tail_root', 'tail_middle'), id=0, color=[221, 94, 86]),
    #     1: dict(link=('tail_middle', 'tail_end'), id=1, color=[221, 94, 86]),
    #     2: dict(link=('tail_root', 'left_foot'), id=2, color=[210, 220, 88]),
    #     # 3: dict(link=('body_middle', 'left_foot'), id=3, color=[210, 220, 88]),
    #     4: dict(link=('tail_root', 'right_foot'), id=4, color=[98, 201, 211]),
    #     # 5: dict(link=('body_middle', 'right_foot'), id=5, color=[98, 201, 211]),
    #     6: dict(link=('tail_root', 'body_middle'), id=6, color=[221, 94, 86]),
    #     7: dict(link=('body_middle', 'neck'), id=7, color=[221, 94, 86]),
    #     8: dict(link=('neck', 'nose'), id=8, color=[221, 94, 86]),
    #     9: dict(link=('nose', 'left_ear_tip'), id=9, color=[92, 94, 170]),
    #     10: dict(link=('nose', 'right_ear_tip'), id=10, color=[92, 94, 170]),
    #     11: dict(link=('neck', 'left_paw_end'), id=11, color=[187, 97, 166]),
    #     # 12: dict(link=('body_middle', 'left_paw_end'), id=12, color=[187, 97, 166]),
    #     13: dict(link=('neck', 'right_paw_end'), id=13, color=[109, 192, 91]),
    #     # 14: dict(link=('body_middle', 'right_paw_end'), id=14, color=[109, 192, 91]),
    # },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        # 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.
    ],

    # Note: The original paper did not provide enough information about
    # the sigmas. We modified from 'https://github.com/cocodataset/'
    # 'cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py#L523'
    sigmas=[
        0.02, 0.02, 0.02, 0.1,
        0.15, 0.02, 0.02, 0.02,
        0.03, 0.03, 0.03, 0.03]
)