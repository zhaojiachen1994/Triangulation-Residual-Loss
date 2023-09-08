
import copy
import json
import os.path as osp
import pickle
import tempfile
from collections import OrderedDict

import mmcv
import numpy as np
# from icecream import ic
from mmcv import Config
from mmpose.core.evaluation import keypoint_mpjpe
from mmpose.datasets.builder import DATASETS
from mmpose.datasets.datasets.base import Kpt3dMviewRgbImgDirectDataset
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval


@DATASETS.register_module()
class MouseDannce3dDataset(Kpt3dMviewRgbImgDirectDataset):
    ALLOWED_METRICS = {'mpjpe'}

    def __init__(self,
                 ann_file,
                 ann_3d_file,
                 cam_file,
                 cam_ids,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False,
                 coco_style=True):
        """
        Args:
            ann_file: the 2d keypoint annotation file in coco form
            ann_3d_file: the 3d keypoint annotation file
            cam_file: camera parameter file
            cam_ids:
            img_prefix:
            data_cfg:
            pipeline:
            dataset_info:
            test_mode:
            coco_style:
        """

        if dataset_info is None:
            cfg = Config.fromfile('configs/_base_/datasets/mouse_datasets/mouse_dannce_p22.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode
        )
        self.cam_ids = cam_ids
        self.num_used_cams = len(cam_ids)
        self.ann_3d_file = ann_3d_file
        self.ann_info['use_different_joint_weights'] = False
        self.use_nms = data_cfg.get('use_nms', True)
        self.soft_nms = data_cfg['soft_nms']
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']
        self.data_cfg = data_cfg

        if coco_style:
            self.coco = COCO(ann_file)
            if 'categories' in self.coco.dataset:
                cats = [
                    cat['name']
                    for cat in self.coco.loadCats(self.coco.getCatIds())
                ]
                self.classes = ['__background__'] + cats
                self.num_classes = len(self.classes)
                self._class_to_ind = dict(
                    zip(self.classes, range(self.num_classes)))
                self._class_to_coco_ind = dict(
                    zip(cats, self.coco.getCatIds()))
                self._coco_ind_to_class_ind = dict(
                    (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                    for cls in self.classes[1:])
            self.img_ids = self.coco.getImgIds()
            self.num_images = len(self.img_ids)
            self.id2name, self.name2id = self._get_mapping_id_name(
                self.coco.imgs)

        self.cameras = self._get_cam(cam_file)
        self.db = self._get_db(data_cfg)

        # get the 3d keypoint ground truth, here written as joints_4d to match the mmpose denotation
        self.joints_4d_data = self._get_joints_3d(data_cfg)

    def _get_cam(self, calib):
        """Get camera parameters.
        Returns: Camera parameters.
        """
        # camera_file = "D:/BaiduNetdiskDownload/coco_data/cams.pkl"
        with open(calib, 'rb') as f:
            data = pickle.load(f)
        cameras = {}
        for i, cam in enumerate(data):
            cameras[i] = {}
            cameras[i]['K'] = cam['K'].T
            cameras[i]['R'] = cam['R'].T
            cameras[i]['T'] = cam['T']
            # cameras[i]['dist_coeff'] = cam['dist_coeff']
            cameras[i]['k'] = [cam['dist_coeff'][0],
                               cam['dist_coeff'][1],
                               cam['dist_coeff'][4]]
            cameras[i]['p'] = [cam['dist_coeff'][2],
                               cam['dist_coeff'][3]]
        return cameras

    def _get_db(self, data_cfg):
        """get the database"""
        gt_db = []
        for img_id in self.img_ids:
            img_ann = self.coco.loadImgs(img_id)[0]

            width = img_ann['width']
            height = img_ann['height']
            # num_joints = self.ann_info['num_joints']
            num_joints = data_cfg['num_joints']
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            objs = self.coco.loadAnns(ann_ids)
            # sanitize bboxes
            valid_objs = []
            for obj in objs:
                if 'bbox' not in obj:
                    continue
                x, y, w, h = obj['bbox']
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(width - 1, x1 + max(0, w))
                y2 = min(height - 1, y1 + max(0, h))
                if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                    obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                    valid_objs.append(obj)
            objs = valid_objs

            bbox_id = 0
            rec = []
            for obj in objs:
                if 'keypoints' not in obj:
                    continue
                if max(obj['keypoints']) == 0:
                    continue
                if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
                    continue

                # get the 2d keypoint ground truth, here written as joints_3d to match the mmpose denotation
                joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)
                keypoints = np.array(obj['keypoints']).reshape(-1, 3)
                joints_3d[:, :2] = keypoints[data_cfg['dataset_channel'], :2]
                joints_3d_visible[:, :2] = np.minimum(1, keypoints[data_cfg['dataset_channel'], 2:3])

                image_file = osp.join(self.img_prefix, self.id2name[img_id])
                rec.append({
                    'image_file': image_file,
                    'joints_3d': joints_3d,
                    'joints_3d_visible': joints_3d_visible,
                    'scene_id': obj['scene_id'],
                    'cam_id': obj['cam_id'],
                    'bbox': obj['clean_bbox'][:4],
                    'rotation': 0,
                    'dataset': self.dataset_name,
                    'bbox_score': 1,
                    'bbox_id': bbox_id
                })
                bbox_id = bbox_id + 1
            gt_db.extend(rec)
        return gt_db

    def _get_joints_3d(self, data_cfg):
        """load the ground truth 3d keypoint, annoted as 4d in outer space"""
        with open(self.ann_3d_file, 'rb') as f:
            data = json.load(f)
        return data

    def __getitem__(self, idx):
        """Get the sample by a given index"""
        results = {}
        for i, c in enumerate(self.cam_ids):
            result = copy.deepcopy(self.db[self.num_cameras * idx + c])
            scene_id = self.db[self.num_cameras * idx + c]['scene_id']
            result['ann_info'] = self.ann_info
            result['camera_0'] = self.cameras[c]
            result['camera'] = copy.deepcopy(self.cameras[c])
            result['joints_4d'] = \
                np.array(self.joints_4d_data[str(scene_id)]['joints_3d'])[self.data_cfg['dataset_channel'], :]
            result['joints_4d_visible'] = \
                np.array(self.joints_4d_data[str(scene_id)]['joints_3d_visible'])[self.data_cfg['dataset_channel']]
            # dummy label
            result['label'] = 0,
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

    def _do_python_keypoint_eval(self, res_file):
        """helper function for evaluate, Keypoint evaluation using COCOAPI."""
        coco_det = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_det, 'keypoints', self.sigmas)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """helper function for evaluate, sort kpts and remove the repeated ones."""
        for img_id, persons in kpts.items():
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts


if __name__ == "__main__":
    print("---")
