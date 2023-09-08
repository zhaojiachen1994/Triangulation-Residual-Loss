import warnings

import torch
from icecream import ic

from .triangnet import TriangNet
from mmpose.models import builder
from mmpose.models.builder import POSENETS
try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16

@POSENETS.register_module()
class DATriangNet(TriangNet):
    """
    The domain adversarial Triangulation net
    """
    def __init__(self,
                 backbone,
                 keypoint_head,
                 domain_discriminator=None,
                 triangulate_head=None,
                 score_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(backbone, keypoint_head, triangulate_head, score_head, train_cfg, test_cfg, pretrained)
        if domain_discriminator is not None:
            ic(domain_discriminator)
            self.domain_discriminator = builder.build_head(domain_discriminator)

    @property
    def with_domain_discriminator(self):
        return hasattr(self, 'domain_discriminator')

    def forward(self, source_data=None, target_data=None, return_loss=True, return_heatmap=False):
        if return_loss:
            return self.forward_train(source_data, target_data)
        else:
            return self.forward_test(**target_data, return_heatmap=return_heatmap)

    def forward_train(self, source_data, target_data):
        # source data is from 2d source domain
        source_img = source_data['img']  # [num_source_img, 3, 256, 256]
        source_heatmap_gt = source_data['target']  # [num_source_img, num_joint, 64, 64]
        source_heatmap_weight = source_data['target_weight']  # [num_source_img, num_joint],
        # source_img_metas = source_data['img_metas']
        num_source_img = source_img.shape[0]

        # target data is from multiview target domain
        target_img = target_data['img']  # [num_scene, num_cams, 3, 256, 256]
        proj_mat = target_data['proj_mat']
        target_img = target_img.reshape(-1, *target_img.shape[2:])  # [num_scene*num_cams, 3, 256, 256]
        num_target_img = target_img.shape[0]
        # target_heatmap_gt = target_data['target']  # [num_scene, num_cams, num_joint, 64, 64]
        # target_heatmap_weight = target_data['target_weight']
        # joints_gt = target_data['joints_4d'] # the 3d joint ground truth
        # joints_visible = target_data['joints_4d_visible']
        # num_target_scene = target_img.shape[0]
        # num_cams = target_img.shape[1]
        # target_heatmap_gt = target_heatmap_gt.reshape(-1, *target_heatmap_gt.shape[2:])
        # target_heatmap_weight = target_heatmap_weight.reshape(-1, *target_heatmap_weight.shape[2:])
        img = torch.concat([source_img, target_img], dim=0)  # [num_source_img+num_scene*num_cams, 3, 256, 256]

        # forward backbone
        hidden_features = self.backbone(img)

        losses = dict()
        # forward domain discriminator
        if self.with_domain_discriminator:
            d_s, d_t, d_label_s, d_label_t = \
                self.domain_discriminator(hidden_features, num_source_img, num_target_img)
            dd_losses = self.domain_discriminator.get_loss(d_s, d_t, d_label_s, d_label_t)
            losses.update(dd_losses)
            dd_accuracy = self.domain_discriminator.get_accuracy(d_s, d_t, d_label_s, d_label_t)
            losses.update(dd_accuracy)

        # forward keypoint head
        heatmap = self.keypoint_head(hidden_features) # [num_source_img+num_scene*num_cams, num_joints, 64, 64]
        source_heatmap_pred = heatmap[:num_source_img]
        target_heatmap_pred = heatmap[num_source_img:]

        # if use source 2d supervised loss
        if self.train_cfg.get('source_2d_sup_loss', False):
            keypoint2d_losses = self.keypoint_head.get_loss(source_heatmap_pred,
                                                            source_heatmap_gt,
                                                            source_heatmap_weight)
            losses.update(keypoint2d_losses)
            keypoint_accuracy = self.keypoint_head.get_accuracy(source_heatmap_pred,
                                                                source_heatmap_gt,
                                                                source_heatmap_weight)
            losses.update(keypoint_accuracy)

        # forward score_head and triangulate_head
        if self.with_score_head:
            scores = self.score_head(hidden_features[0][-num_target_img:])  # [num_scene*num_cams, num_joints]
        else:
            scores = torch.ones([num_target_img, source_heatmap_gt.shape[1]], dtype=torch.float32, device=target.device)
        kpt_3d_pred, res_triang, kp_2d_croped, _, _ = \
                        self.triangulate_head(target_heatmap_pred, proj_mat, scores)

        # if use target 3d unsupervised loss
        if self.with_triangulate_head and self.train_cfg.get('target_3d_unsup_loss'):
            unsup_3d_loss = self.triangulate_head.get_unSup_loss(res_triang)
            losses.update(unsup_3d_loss)
        return losses



