import warnings

import torch
from icecream import ic
from .triangnet import TriangNet
from mmpose.models.builder import POSENETS

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16

@POSENETS.register_module()
class CDTriangNet(TriangNet):
    def __init__(self,
                 backbone,
                 keypoint_head,
                 triangulate_head=None,
                 score_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(backbone, keypoint_head, triangulate_head, score_head, train_cfg, test_cfg, pretrained)

    @auto_fp16(apply_to=('img',))
    def forward(self, source_data=None, target_data=None, return_loss=True, return_heatmap=False):
        if return_loss:
            return self.forward_train(source_data, target_data)
        else:
            return self.forward_test(**target_data, return_heatmap=return_heatmap)

    def forward_train(self, source_data, target_data):

        # source data is from 2d source domain
        source_img = source_data['img'] # [num_source_img, 3, 256, 256]
        source_heatmap_gt = source_data['target'] # [num_source_img, num_joint, 64, 64]
        source_heatmap_weight = source_data['target_weight']    # [num_source_img, num_joint]
        # source_img_metas = source_data['img_metas']
        num_source_img = source_img.shape[0]


        # target data is from multiview target domain
        target_img = target_data['img'] # [num_scene, num_cams, 3, 256, 256]
        # target_heatmap_gt = target_data['target']   # [num_scene, num_cams, num_joint, 64, 64]
        # target_heatmap_weight = target_data['target_weight']
        proj_mat = target_data['proj_mat']
        joints_4d = target_data['joints_4d'] # the 3d joint ground truth
        joints_4d_visible = target_data['joints_4d_visible']

        # num_target_scene = target_img.shape[0]
        # num_cams = target_img.shape[1]

        # reshape the target domain data
        target_img = target_img.reshape(-1, *target_img.shape[2:])  # [num_scene*num_cams, 3, 256, 256]
        num_target_img = target_img.shape[0]
        # target_heatmap_gt = target_heatmap_gt.reshape(-1, *target_heatmap_gt.shape[2:])
        # target_heatmap_weight = target_heatmap_weight.reshape(-1, *target_heatmap_weight.shape[2:])



        # concat the source domain and target domain image data
        img = torch.concat([source_img, target_img], dim=0) # [num_source_img+num_scene*num_cams, 3, 256, 256]
        # heatmap_gt = torch.concat([source_heatmap_gt, target_heatmap_gt], dim=0)
        # heatmap_weight = torch.concat([source_heatmap_weight, target_heatmap_weight])
        # ic(img.shape)
        # source target forward bachbone and keypoint head
        hidden_features = self.backbone(img)[0]
        heatmap = self.keypoint_head(hidden_features)   # [num_source_img+num_scene*num_cams, num_joints, 64, 64]


        hidden_features_target = hidden_features[num_source_img:]   # if hidden_features is list, use hidden_features[0][num_scoure_img:]
        # ic(hidden_features_target.shape)
        source_heatmap_pred = heatmap[:num_source_img]
        target_heatmap_pred = heatmap[num_source_img:]

        if self.with_score_head:
            scores = self.score_head(hidden_features_target)  # [num_scene*num_cams, num_joints]
        else:
            scores = torch.ones([num_target_img, source_heatmap_gt.shape[1]], dtype=torch.float32,
                                device=heatmap.device)
        # print(scores)
        if self.train_cfg['score_thr'] is not None:
            scores = torch.clamp(scores, min=self.train_cfg['score_thr'][0],
                                 max=self.train_cfg['score_thr'][1])
        # ic(scores)
        # scores = torch.clamp(scores, min=0.2, max=0.8)
        if self.with_triangulate_head:
            kpt_3d_pred, res_triang, kp_2d_croped, _, _ = \
                self.triangulate_head(target_heatmap_pred, proj_mat, scores)

        losses = dict()
        if self.train_cfg.get('source_2d_sup_loss', False):
            keypoint2d_losses = self.keypoint_head.get_loss(source_heatmap_pred,
                                                            source_heatmap_gt,
                                                            source_heatmap_weight)
            losses.update(keypoint2d_losses)
            keypoint_accuracy = self.keypoint_head.get_accuracy(source_heatmap_pred,
                                                                source_heatmap_gt,
                                                                source_heatmap_weight)
            losses.update(keypoint_accuracy)



        if self.with_triangulate_head and self.train_cfg.get('target_3d_unsup_loss'):
            unsup_3d_loss = self.triangulate_head.get_unSup_loss(res_triang)
            losses.update(unsup_3d_loss)

        if self.with_triangulate_head and self.train_cfg.get('target_3d_sup_loss'):
            sup_3d_loss = self.triangulate_head.get_sup_loss(kpt_3d_pred, joints_4d, joints_4d_visible)
            losses.update(sup_3d_loss)
        return losses















