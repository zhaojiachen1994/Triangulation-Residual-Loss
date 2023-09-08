import cv2
import numpy as np
import torch
from PIL import Image
from mmpose.datasets import PIPELINES


@PIPELINES.register_module()
class DummyTransform:
    def __call__(self, results):
        results['dummy'] = True
        return results


@PIPELINES.register_module()
class SquareBbox:
    """Makes square bbox from any bbox by stretching of minimal length side
    bbox is defined by xywh

    Required key: 'bbox', 'ann_info'
    Modified key: 'bbox',

    return bbox by xyxy

    """
    def __init__(self, format='xywh'):
        self.format=format

    def __call__(self, results):
        bbox = results['bbox']

        if self.format == 'xyxy':    # the det model output
            left, upper, right, lower = bbox
            width = right - left
            height = lower - upper
        elif self.format == 'xywh':  # the dataset annotation format
            left, upper, width, height = bbox
            right = left + width
            lower = upper + height

        if width > height:
            y_center = (upper + lower) // 2
            upper = y_center - width // 2
            lower = upper + width
            height = width
        else:
            x_center = (left + right) // 2
            left = x_center - height // 2
            right = left + height
            width = height

        results['bbox'] = [left, upper, right, lower]
        return results


@PIPELINES.register_module()
class CropImage:
    """Crops area from image specified as bbox. Always returns area of size as bbox filling missing parts with zeros

        Required key: 'bbox', 'img', 'ann_info', 'joint_3d'; 'camera' if update_camera is True
        Modified key: 'img', add 'bbox_offset'

        Args:
            image numpy array of shape (height, width, 3): input image
            bbox tuple of size 4: input bbox (left, upper, right, lower)
            joint_3d: [num_joints, 3], [x, y, 0]

        Returns:
            cropped_image numpy array of shape (height, width, 3): resulting cropped image
    """
    def __init__(self, update_camera=False, update_gt=True):
        self.update_camera = update_camera
        self.update_gt = update_gt

    def __call__(self, results):
        image_pil = Image.fromarray(results['img'])
        image_pil = image_pil.crop(results['bbox'])
        results['img'] = np.asarray(image_pil)

        # update the ground truth keypoint coord
        left = results['bbox'][0]
        upper = results['bbox'][1]

        if self.update_gt:
            joint_3d = results['joints_3d'] # 2d indeed
            joint_3d[:, 0] = joint_3d[:, 0] - left
            joint_3d[:, 1] = joint_3d[:, 1] - upper
            results['joints_3d'] = joint_3d

        results['bbox_offset'] = results['bbox'][:2]

        if self.update_camera:
            camera = results['camera_0']
            left, upper, right, lower = results['bbox']
            cx, cy = camera['K'][0, 2], camera['K'][1, 2]

            new_cx = cx - left
            new_cy = cy - upper
            results['camera']['K'][0, 2], results['camera']['K'][1, 2] = new_cx, new_cy

        return results


@PIPELINES.register_module()
class ResizeImage:
    """
    resize the croped box into input image size
    """
    def __init__(self, update_camera=False, update_gt=True):
        self.update_camera = update_camera
        self.update_gt = update_gt

    def __call__(self, results):
        # ic(results['image_file'], results['cam_idx'], results['frame_idx'],results['camera']['K'] )
        img = results['img']
        [height_old, width_old, _] = img.shape

        [new_height, new_width] = results['ann_info']['image_size']
        results['img'] = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # update the ground truth 2d keypoint coord
        if self.update_gt:
            results['joints_3d'][:, 0] = results['joints_3d'][:, 0] * (new_width / width_old)
            results['joints_3d'][:, 1] = results['joints_3d'][:, 1] * (new_width / width_old)


        # save the resize ratio
        results['resize_ratio'] = new_width / width_old

        if self.update_camera:
            camera = results['camera']
            fx, fy, cx, cy = camera['K'][0, 0], camera['K'][1, 1], camera['K'][0, 2], camera['K'][1, 2]
            new_fx = fx * (new_width / width_old)
            new_fy = fy * (new_height / height_old)
            new_cx = cx * (new_width / width_old)
            new_cy = cy * (new_height / height_old)
            results['camera']['K'][0, 0], \
            results['camera']['K'][1, 1], \
            results['camera']['K'][0, 2], \
            results['camera']['K'][1, 2] = new_fx, new_fy, new_cx, new_cy
        # ic(results['image_file'], results['cam_idx'], results['frame_idx'], results['camera']['K'])

        return results


@PIPELINES.register_module()
class GroupCams:
    """
        Required key: 'img', 'target', 'target_weight'
        Modified key: 'img'
    """
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            if isinstance(results[key][0], torch.Tensor):
                results[key] = torch.stack(results[key])
            elif isinstance(results[key][0], np.ndarray):
                results[key] = np.stack(results[key])
        return results


@PIPELINES.register_module()
class ComputeProjMatric:
    """
    compute the project matrics for camera based on the camera cameras
    Required key: 'camera'
    Added key: 'proj_matrics' [3, 4]
    """
    def __call__(self, results):
        n_cams = len(results['camera'])
        proj_metric = np.zeros([n_cams, 3, 4])
        # ic(results['camera']['K'].shape)
        # ic(results['camera']['R'].shape)
        # ic(results['camera']['T'].shape)
        # ic(np.hstack([results['camera'][0]['R'], np.transpose(results['camera'][0]['T'])]))
        # ic(results['camera']['K'])
        proj_metric = results['camera']['K'].dot(np.hstack([results['camera']['R'],
                                                            results['camera']['T'].reshape([3, 1])]))
        # proj_metric = [params['K'].dot(np.hstack([params['R'], np.transpose(params['T'])])) for params in
        #                results['camera']]
        results['proj_mat'] = proj_metric
        # ic(results['image_file'], results['cam_idx'], results['frame_idx'], results['proj_mat'], results['camera']['K'])
        return results
