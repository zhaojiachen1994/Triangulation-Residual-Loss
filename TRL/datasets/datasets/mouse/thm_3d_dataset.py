import copy
import json
import os.path as osp
import tempfile
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv import Config
from xtcocotools.coco import COCO

from mmpose.core.evaluation import keypoint_mpjpe
from mmpose.datasets.builder import DATASETS
from mmpose.datasets.datasets.base import Kpt3dMviewRgbImgDirectDataset


@DATASETS.register_module()
class THM3dDatasetMview(Kpt3dMviewRgbImgDirectDataset):
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
                 ):
        if dataset_info is None:
            cfg = Config.fromfile("configs/_base_/mouse_datasets/mouse_one_1229.py")
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
        self.data_cfg = data_cfg
        self.num_cameras = data_cfg['num_cameras']
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        self.num_images = len(self.img_ids)
        self.id2name, self.name2id = self._get_mapping_id_name(
            self.coco.imgs)
        self.db = self._get_db(data_cfg)
        self.cams_params = self._get_cam_params(cam_file)
        # get the 3d keypoint ground truth, here written as joints_4d to match the mmpose denotation
        self.joints_4d, self.joints_4d_visible, _ = self._get_joints_3d(self.ann_3d_file, data_cfg)

    def __len__(self):
        return int(self.num_images / self.num_cameras)

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

            # only one mouse in this dataset
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
                'cam_idx': image_file[-16:-12],
                'scene_id': obj['scene_id'],
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'bbox': obj['clean_bbox'][:4],
                'rotation': 0,
                'dataset': self.dataset_name,
                'bbox_score': 1,
                'bbox_id': 0,
            }
            gt_db.append(rec)
        return gt_db

    def _get_cam_params(self, calib):
        with open(calib, 'r') as f:
            cameras_params = json.load(f)
        new_cameras_params = {}
        for k in cameras_params.keys():
            new_cameras_params[k] = {}
            new_cameras_params[k]['K'] = np.array(cameras_params[k]['K']).reshape([3, 3])
            new_cameras_params[k]['R'] = np.array(cameras_params[k]['R'])
            new_cameras_params[k]['T'] = np.array(cameras_params[k]['T'])
        return new_cameras_params

    def _get_joints_3d(self, ann_3d_file, data_cfg):
        """load the ground truth 3d keypoint, annoted as 4d in outer space"""
        with open(ann_3d_file, 'rb') as f:
            data = json.load(f)
        data = np.array(data['joint_3d'])
        data = data[:, data_cfg['dataset_channel'], :]

        [num_sample, num_joints, _] = data.shape
        data[np.isnan(data)] = 0.0

        # joints_3d
        joints_3d = np.zeros_like(data, dtype=np.float32)
        joints_3d[:] = data[:]

        # joints_3d_visible
        joints_3d_visible = np.ones_like(data, dtype=np.float32)
        joints_3d_visible[joints_3d == 0] = 0.0
        joints_3d_visible = joints_3d_visible.reshape([num_sample, num_joints, 3])

        roots_3d = data[:, 5, :]  # body_middle as root here
        return joints_3d, joints_3d_visible, roots_3d

    def __getitem__(self, idx):
        """Get the sample by a given index"""
        results = {}
        for c in range(self.num_cameras):
            result = copy.deepcopy(self.db[self.num_cameras * idx + c])
            scene_id = self.db[self.num_cameras * idx + c]['scene_id']
            result['ann_info'] = self.ann_info
            result['camera_0'] = self.cams_params[result['cam_idx']]  # the original camera
            result['camera'] = copy.deepcopy(self.cams_params[result['cam_idx']])  # updated camera in pipeline
            result['joints_4d'] = self.joints_4d[scene_id]
            result['joints_4d_visible'] = self.joints_4d_visible[scene_id]
            results[c] = result
        return self.pipeline(results)

    def evaluate(self, results, res_folder=None, metric='mpjpe', **kwargs):
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
            gts.append(gt)
            preds.append(pred)
            masks.append(mask)
        preds = np.stack(preds)
        gts = np.stack(gts)  # [num_samples, ]
        masks = np.stack(masks) > 0  # [num_samples, num_joints]
        masks = masks[:, :, 0]
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

    def _write_coco_keypoint_results(self, keypoints, res_file):
        """helper function for evaluate, Write results into a json file."""
        data_pack = [{
            'cat_id': self._class_to_coco_ind[cls],
            'cls_ind': cls_ind,
            'cls': cls,
            'ann_type': 'keypoints',
            'keypoints': keypoints
        } for cls_ind, cls in enumerate(self.classes)
            if not cls == '__background__']

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])

        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        """helper function for evaluate, Get coco keypoint results."""
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            key_points = _key_points.reshape(-1,
                                             self.ann_info['num_joints'] * 3)

            result = [{
                'image_id': img_kpt['image_id'],
                'category_id': cat_id,
                'keypoints': key_point.tolist(),
                'score': float(img_kpt['score']),
                'center': img_kpt['center'].tolist(),
                'scale': img_kpt['scale'].tolist()
            } for img_kpt, key_point in zip(img_kpts, key_points)]

            cat_results.extend(result)

        return cat_results
