dataset_info = dict(
    dataset_name='p9',
    paper_info=dict(
        author='Cristina Segalin',
        title='The Mouse Action Recognition System (MARS) software pipeline for automated analysis of social behaviors in mice',
        container='eLife',
        year='2021',
        homepage=''),

    keypoint_info={
        0: dict(name='nose', id=2, color=[92, 94, 170], type='upper', swap=''),
        1: dict(name='left_ear', id=1, color=[92, 94, 170], type='upper', swap='right_ear'),
        2: dict(name='right_ear', id=2, color=[92, 94, 170], type='upper', swap='left_ear'),
        3: dict(name='neck', id=3, color=[221, 94, 86], type='upper', swap=''),
        4: dict(name='tail_root', id=4, color=[221, 94, 86], type='upper', swap=''),
        5: dict(name='left_paw_end', id=5, color=[187, 97, 166], type='upper', swap='right_paw_end'),
        6: dict(name='right_paw_end', id=6, color=[109, 192, 91], type='upper', swap='left_paw_end'),
        7: dict(name='left_foot', id=10, color=[210, 220, 88], type='upper', swap='right_foot'),
        8: dict(name='right_foot', id=11, color=[98, 201, 211], type='lower', swap='left_foot'),
    },
    skeleton_info={
        0: dict(link=('neck', 'nose'), id=0, color=[221, 94, 86]),
        1: dict(link=('nose', 'left_ear'), id=1, color=[92, 94, 170]),
        2: dict(link=('nose', 'right_ear'), id=2, color=[92, 94, 170]),
        3: dict(link=('tail_root', 'neck'), id=3, color=[221, 94, 86]),
        4: dict(link=('neck', 'left_paw_end'), id=4, color=[187, 97, 166]),
        5: dict(link=('neck', 'right_paw_end'), id=5, color=[109, 192, 91]),
        6: dict(link=('tail_root', 'left_foot'), id=2, color=[210, 220, 88]),
        7: dict(link=('tail_root', 'right_foot'), id=4, color=[98, 201, 211]),
    },
    joint_weights=[1., 1., 1., 1., 1., 1., 1., 1., 1.],
    sigmas=[
        0.02, 0.02, 0.02, 0.1, 0.02,
        0.03, 0.03, 0.03, 0.03]
)
