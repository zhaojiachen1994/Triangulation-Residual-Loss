# my h36m dataset class

import copy
import json
import os.path as osp
import tempfile
from collections import OrderedDict
from icecream import ic
import mmcv
import numpy as np
from mmcv import Config
from xtcocotools.coco import COCO

from mmpose.core.evaluation import keypoint_mpjpe
# from mmpose.datasets import DATASETS
# from TRL.datasets import DATASETS
from mmpose.datasets.builder import DATASETS
from mmpose.datasets.datasets.base import Kpt3dMviewRgbImgDirectDataset


@DATASETS.register_module()
class Body3DH36MMviewDataset(Kpt3dMviewRgbImgDirectDataset):
    # metric
    ALLOWED_METRICS = {'mpjpe', 'p-mpjpe', 'n-mpjpe'}

    def __init__(self,
                 ann_file,
                 ann_3d_file,
                 cam_file,
                 img_prefix,
                 data_cfg,
                 pipeline=None,
                 dataset_info=None,
                 test_mode=False,
                 coco_style=True
                 ):
        if dataset_info is None:
            cfg = Config.fromfile("configs/_base_/datasets/h36m.py")
            dataset_info = cfg._cfg_dict['dataset_info']
        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode,
        )
        self.ann_3d_file = ann_3d_file
        self.ann_info['use_different_joint_weights'] = False
        self.use_nms = data_cfg.get('use_nms', True)
        self.soft_nms = data_cfg['soft_nms']
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']
        self.num_joints = data_cfg['num_joints']
        self.num_cameras = 4
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        self.num_images = len(self.img_ids)
        self.data_cfg = data_cfg

        self.id2name, self.name2id = self._get_mapping_id_name(
            self.coco.imgs)
        self.db = self._get_db(data_cfg)
        self.cams_params = self._get_cam_params(cam_file)
        self.joints_global = self._get_joints_3d(ann_3d_file)

    def __len__(self):
        return int(self.num_images / 4)

    def _get_db(self, data_cfg):
        """get the database"""
        gt_db = []
        for img_id in self.img_ids:
            img_ann = self.coco.loadImgs(img_id)[0]
            width = img_ann['width']
            height = img_ann['height']
            num_joints = self.ann_info['num_joints']
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            objs = self.coco.loadAnns(ann_ids)

            # only a person in h36m
            obj = objs[0]
            x, y, w, h = obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w))
            y2 = min(height - 1, y1 + max(0, h))
            if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]

            # get the 2d keypoint ground truth, here written as joints_3d to match the mmpose denotation
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)
            keypoints = np.array(obj['keypoints']).reshape(-1, 3)
            joints_3d[:, :2] = keypoints[data_cfg['dataset_channel'], :2]
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[data_cfg['dataset_channel'], 2:3])
            image_file = osp.join(self.img_prefix, self.id2name[img_id])

            rec = {
                'image_file': image_file,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'bbox': obj['clean_bbox'][:4],
                'rotation': 0,
                'dataset': self.dataset_name,
                'bbox_score': 1,
                'bbox_id': 0,
                'subject': img_ann['subject'],
                'action_idx': img_ann['action_idx'],
                'subaction_idx': img_ann['subaction_idx'],
                'cam_idx': img_ann['cam_idx'],
                'frame_idx': img_ann['frame_idx']
            }
            gt_db.append(rec)
        return gt_db


    # def _get_cam_params(self, calib):
    #     with open(calib, 'r') as f:
    #         cameras_params = json.load(f)
    #     new_cameras_params = {}
    #     for key in cameras_params.keys():
    #         param = cameras_params[key]
    #         new_cameras_params[key] = {}
    #         R = np.array(param['R'], dtype=np.float32)
    #         T = np.array(param['T'], dtype=np.float32).reshape(3, 1)
    #         f = np.array(param['f'], dtype=np.float32).reshape([2, 1])
    #         c = np.array(param['c'], dtype=np.float32).reshape([2, 1])
    #         K = np.concatenate((np.diagflat(f), c), axis=-1).T
    #         K = np.hstack([K, np.array([0.0, 0.0, 1.0]).reshape([3, 1])]).T
    #
    #         new_cameras_params[key]['R'] = R
    #         new_cameras_params[key]['T'] = T
    #         new_cameras_params[key]['K'] = K
    #     return new_cameras_params

    def _get_cam_params(self, calib):   # for the anliang's data
        with open(calib, 'r') as f:
            cameras_params = json.load(f)
        new_cameras_params = {}
        for s in cameras_params.keys():
            new_cameras_params[s] = {}
            for i in [1, 2, 3, 4]:
                new_cameras_params[s][i] = {}
                param = cameras_params[s][str(i)]
                R = np.array(param['R'], dtype=np.float32)
                T = np.array(param['t'], dtype=np.float32).reshape(3, 1)
                f = np.array(param['f'], dtype=np.float32).reshape([2, 1])
                c = np.array(param['c'], dtype=np.float32).reshape([2, 1])
                K = np.concatenate((np.diagflat(f), c), axis=-1).T
                K = np.hstack([K, np.array([0.0, 0.0, 1.0]).reshape([3, 1])]).T

                new_cameras_params[s][i]['R'] = R
                new_cameras_params[s][i]['T'] = T
                new_cameras_params[s][i]['K'] = K
            # extrinsics = np.hstack([R, T])
            # projection = K.dot(extrinsics)
            # projections[i] = projection

        return new_cameras_params

    def _get_joints_3d(self, ann_3d_file):
        """load the ground truth 3d keypoint, annoted as 4d in outer space"""
        with open(ann_3d_file, 'rb') as f:
            data = json.load(f)
        return data

    def __getitem__(self, idx):
        """Get the sample by a given index"""
        results = {}
        for c in range(self.num_cameras):
            result = copy.deepcopy(self.db[self.num_cameras * idx + c])
            result['ann_info'] = self.ann_info
            result['camera_0'] = self.cams_params[str(result['subject'])][result['cam_idx']]  # the original camera
            # result['camera_0'] = self.cams_params[f"{result['subject']}-{result['cam_idx']}"]
            result['camera'] = copy.deepcopy(result['camera_0'])  # updated camera in pipeline
            result['updated_cam'] = False
            # get the global 3d joint
            result['joints_4d'] = np.array(self.joints_global[str(idx)])[self.data_cfg['dataset_channel']]
            result['joints_4d_visible'] = np.ones(self.data_cfg['num_joints'])
            results[c] = result

        return self.pipeline(results)

    def get_gt_global(self, subject, action_idx, subaction_idx, frame_idx):
        """get the ground truth 3d joints based on the img_metas used in evaluate"""
        # subject = img_metas['subject']
        # action_idx = img_metas['action_idx']
        # subaction_idx = img_metas['subaction_idx']
        # frame_idx = img_metas['frame_idx']
        gt_global = self.joints_global[str(action_idx)][str(subaction_idx)][str(frame_idx)]
        return gt_global

    def evaluate(self, results, res_folder=None, metric='mpjpe', *args, **kwargs):
        metrics = metric if isinstance(metric, list) else [metric]
        for _metric in metrics:
            if _metric not in self.ALLOWED_METRICS:
                raise ValueError(
                    f'Unsupported metric "{_metric}" for human3.6 dataset.'
                    f'Supported metrics are {self.ALLOWED_METRICS}')

        if res_folder is not None:
            tmp_folder = None
            res_file = osp.join(res_folder, 'result_keypoints.json')
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_keypoints.json')

        kpts = []
        for result in results:  # results contain all batches in test set
            preds = result['preds']
            img_metas = result['img_metas']
            batch_size = len(img_metas)
            for i in range(batch_size):  # result in a batch
                kpts.append({
                    'keypoints': preds[i],
                    'joints_4d': img_metas[i]['joints_4d'],
                    'joints_4d_visible': img_metas[i]['joints_4d_visible'],
                    # 'subject': img_metas[i]['subject'],
                    # 'action_idx': img_metas[i]['action_idx'],
                    # 'subaction_idx': img_metas[i]['subaction_idx'],
                    # 'frame_idx': img_metas[i]['frame_idx'],
                })
        mmcv.dump(kpts, res_file)

        name_value_tuples = []
        for _metric in metrics:
            if _metric == 'mpjpe':
                _nv_tuples = self._report_mpjpe(kpts)
            elif _metric == 'p-mpjpe':
                _nv_tuples = self._report_mpjpe(kpts, mode='p-mpjpe')
            elif _metric == 'n-mpjpe':
                _nv_tuples = self._report_mpjpe(kpts, mode='n-mpjpe')
            else:
                raise NotImplementedError
            name_value_tuples.extend(_nv_tuples)

        if tmp_folder is not None:
            tmp_folder.cleanup()

        return OrderedDict(name_value_tuples)

    def _report_mpjpe(self, keypoint_results, mode='mpjpe'):
        """Cauculate mean per joint position error (MPJPE) or its variants like
        P-MPJPE or N-MPJPE.

        Args:
            keypoint_results (list): Keypoint predictions. See
                'Body3DH36MDataset.evaluate' for details.
            mode (str): Specify mpjpe variants. Supported options are:

                - ``'mpjpe'``: Standard MPJPE.
                - ``'p-mpjpe'``: MPJPE after aligning prediction to groundtruth
                    via a rigid transformation (scale, rotation and
                    translation).
                - ``'n-mpjpe'``: MPJPE after aligning prediction to groundtruth
                    in scale only.
        """
        preds = []
        gts = []
        masks = []
        for idx, result in enumerate(keypoint_results):
            pred = result['keypoints']
            gt = result['joints_4d']
            mask = result['joints_4d_visible'] > 0
            # gt = self.get_gt_global(result['subject'],
            #                         result['action_idx'],
            #                         result['subaction_idx'],
            #                         result['frame_idx'])
            # mask = np.ones(pred.shape[0]) > 0
            gts.append(gt)
            preds.append(pred)
            masks.append(mask)
        preds = np.stack(preds)
        gts = np.stack(gts)  # [num_samples, ]
        masks = np.stack(masks) > 0  # [num_samples, num_joints]
        # ic(preds.shape, gts.shape)
        # ic(masks.shape)

        err_name = mode.upper()
        if mode == 'mpjpe':
            alignment = 'none'
        elif mode == 'p-mpjpe':
            alignment = 'procrustes'
        elif mode == 'n-mpjpe':
            alignment = 'scale'
        else:
            raise ValueError(f'Invalid mode: {mode}')

        error = keypoint_mpjpe(preds, gts, masks, alignment)
        name_value_tuples = [(err_name, error)]
        return name_value_tuples

        #     gt = self.get_gt_global(img_metas)
        #     preds.append(pred)
        #     gts.append(gt)
        # preds = np.stack(preds)
        # gts = np.stack(gts)
        # # masks = np.ones(preds.shape[:2]) > 0

        # name_value_tuples = []
        # for _metric in metrics:
        #     if _metric == 'mpjpe':
        #         _nv_tuples = self._report_mpjpe(preds, gts, masks,)
        #     elif _metric == 'p-mpjpe':
        #         _nv_tuples = self._report_mpjpe(preds, gts, masks, mode='p-mpjpe')
        #     elif _metric == 'n-mpjpe':
        #         _nv_tuples = self._report_mpjpe(preds, gts, masks, mode='n-mpjpe')
        #     else:
        #         raise NotImplementedError
        #     name_value_tuples.extend(_nv_tuples)
        # return OrderedDict(name_value_tuples)


