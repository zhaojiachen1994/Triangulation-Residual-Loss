import numpy as np
import torch
import torch.nn as nn
from icecream import ic
from mmpose.models.builder import build_loss
from mmpose.models import HEADS

def _make_radial_window(width, height, cx, cy, fn, window_width=10.0):
    """
    Returns a grid, where grid[i,j] = fn((i**2 + j**2)**0.5)

    :param width: Width of the grid to return
    :param height: Height of the grid to return
    :param cx: x center
    :param cy: y center
    :param fn: The function to apply
    :return:
    """
    # The length of cx and cy is the number of channels we need
    dev = cx.device
    channels = cx.size(0)

    # Explicitly tile cx and cy, ready for computing the distance matrix below, because pytorch doesn't broadcast very well
    # Make the shape [channels, height, width]
    cx = cx.repeat(height, width, 1).permute(2, 0, 1)
    cy = cy.repeat(height, width, 1).permute(2, 0, 1)

    # Compute a grid where dist[i,j] = (i-cx)**2 + (j-cy)**2, need to view and repeat to tile and make shape [channels, height, width]
    xs = torch.arange(width).view((1, width)).repeat(channels, height, 1).float().to(dev)
    ys = torch.arange(height).view((height, 1)).repeat(channels, 1, width).float().to(dev)
    delta_xs = xs - cx
    delta_ys = ys - cy
    dists = torch.sqrt((delta_ys ** 2) + (delta_xs ** 2))

    # apply the function to the grid and return it
    return fn(dists, window_width)


def _parzen_scalar(delta, width):
    """For reference"""
    del_ovr_wid = math.abs(delta) / width
    if delta <= width / 2.0:
        return 1 - 6 * (del_ovr_wid ** 2) * (1 - del_ovr_wid)
    elif delta <= width:
        return 2 * (1 - del_ovr_wid) ** 3


def _parzen_torch(dists, width):
    """
    A PyTorch version of the parzen window that works a grid of distances about some center point.
    See _parzen_scalar to see the

    :param dists: The grid of distances
    :param window: The width of the parzen window
    :return: A 2d grid, who's values are a (radial) parzen window
    """
    hwidth = width / 2.0
    del_ovr_width = dists / hwidth

    near_mode = (dists <= hwidth / 2.0).float()
    in_tail = ((dists > hwidth / 2.0) * (dists <= hwidth)).float()

    return near_mode * (1 - 6 * (del_ovr_width ** 2) * (1 - del_ovr_width)) \
           + in_tail * (2 * ((1 - del_ovr_width) ** 3))


def _uniform_window(dists, width):
    """
    A (radial) uniform window function
    :param dists: A grid of distances
    :param width: A width for the window
    :return: A 2d grid, who's values are 0 or 1 depending on if it's in the window or not
    """
    hwidth = width / 2.0
    return (dists <= hwidth).float()


def _identity_window(dists, width):
    """
    An "identity window". (I.e. a "window" which when multiplied by, will not change the input).
    """
    return torch.ones(dists.size())


class SoftArgmax1D(torch.nn.Module):
    """
    Implementation of a 1d soft arg-max function as an nn.Module, so that we can differentiate through arg-max operations.
    """

    def __init__(self, base_index=0, step_size=1):
        """
        The "arguments" are base_index, base_index+step_size, base_index+2*step_size, ... and so on for
        arguments at indices 0, 1, 2, ....

        Assumes that the input to this layer will be a batch of 1D tensors (so a 2D tensor).

        :param base_index: Remember a base index for 'indices' for the input
        :param step_size: Step size for 'indices' from the input
        """
        super(SoftArgmax1D, self).__init__()
        self.base_index = base_index
        self.step_size = step_size
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        """
        Compute the forward pass of the 1D soft arg-max function as defined below:

        SoftArgMax(x) = \sum_i (i * softmax(x)_i)

        :param x: The input to the soft arg-max layer
        :return: Output of the soft arg-max layer
        """
        smax = self.softmax(x)
        end_index = self.base_index + x.size()[1] * self.step_size
        indices = torch.arange(start=self.base_index, end=end_index, step=self.step_size)
        return torch.matmul(smax, indices)


class SoftArgmax2D(torch.nn.Module):
    """
    adafuse/lib/models/soft_argmax
    Implementation of a 1d soft arg-max function as an nn.Module, so that we can differentiate through arg-max operations.
    """

    def __init__(self, base_index=0, step_size=1, window_fn=None, window_width=10, softmax_temp=1.0):
        """
        The "arguments" are base_index, base_index+step_size, base_index+2*step_size, ... and so on for
        arguments at indices 0, 1, 2, ....

        Assumes that the input to this layer will be a batch of 3D tensors (so a 4D tensor).
        For input shape (B, C, W, H), we apply softmax across the W and H dimensions.
        We use a softmax, over dim 2, expecting a 3D input, which is created by reshaping the input to (B, C, W*H)
        (This is necessary because true 2D softmax doesn't natively exist in PyTorch...

        :param base_index: Remember a base index for 'indices' for the input
        :param step_size: Step size for 'indices' from the input
        :param window_function: Specify window function, that given some center point produces a window 'landscape'. If
            a window function is specified then before applying "soft argmax" we multiply the input by a window centered
            at the true argmax, to enforce the input to soft argmax to be unimodal. Window function should be specified
            as one of the following options: None, "Parzen", "Uniform"
        :param window_width: How wide do we want the window to be? (If some point is more than width/2 distance from the
            argmax then it will be zeroed out for the soft argmax calculation, unless, window_fn == None)
        """
        super(SoftArgmax2D, self).__init__()
        self.base_index = base_index
        self.step_size = step_size
        self.softmax = torch.nn.Softmax(dim=2)
        self.softmax_temp = softmax_temp
        self.window_type = window_fn
        self.window_width = window_width
        self.window_fn = _identity_window
        if window_fn == "Parzen":
            self.window_fn = _parzen_torch
        elif window_fn == "Uniform":
            self.window_fn = _uniform_window

    def _softmax_2d(self, x, temp):
        """
        For the lack of a true 2D softmax in pytorch, we reshape each image from (C, W, H) to (C, W*H) and then
        apply softmax, and then restore the original shape.

        :param x: A 4D tensor of shape (B, C, W, H) to apply softmax across the W and H dimensions
        :param temp: A scalar temperature to apply as part of the softmax function
        :return: Softmax(x, dims=(2,3))
        """
        B, C, W, H = x.size()
        x_flat = x.view((B, C, W * H)) / temp
        x_softmax = self.softmax(x_flat)
        return x_softmax.view((B, C, W, H))

    def forward(self, x, out_smax=False, hardmax=False):
        """
        Compute the forward pass of the 1D soft arg-max function as defined below:

        SoftArgMax2d(x) = (\sum_i \sum_j (i * softmax2d(x)_ij), \sum_i \sum_j (j * softmax2d(x)_ij))

        :param x: The input to the soft arg-max layer
        :return: [batch_size, 3, channels]
        """
        # Compute windowed softmax
        # Compute windows using a batch_size of "batch_size * channels"
        dev = x.device
        batch_size, channels, height, width = x.size()
        maxv, argmax = torch.max(x.view(batch_size * channels, -1), dim=1)
        argmax_x, argmax_y = torch.remainder(argmax, width).float(), torch.floor(
            torch.div(argmax.float(), float(width)))
        windows = _make_radial_window(width, height, argmax_x, argmax_y, self.window_fn, self.window_width)
        windows = windows.view(batch_size, channels, height, width).to(dev)
        smax = self._softmax_2d(x, self.softmax_temp) * windows
        smax = smax / torch.sum(smax.view(batch_size, channels, -1), dim=2).view(batch_size, channels, 1, 1)

        x_max = argmax_x.view(batch_size, 1, channels)
        y_max = argmax_y.view(batch_size, 1, channels)
        ones = torch.ones_like(x_max)
        xys_max = torch.cat([x_max, y_max, ones], dim=1)
        xys_max = xys_max.view(batch_size, 3, channels)
        # compute x index (sum over y axis, produce with indices and then sum over x axis for the expectation)
        x_end_index = self.base_index + width * self.step_size
        x_indices = torch.arange(start=self.base_index, end=x_end_index, step=self.step_size).type_as(smax)
        x_coords = torch.sum(torch.sum(smax, 2) * x_indices, 2)

        # compute y index (sum over x axis, produce with indices and then sum over y axis for the expectation)
        y_end_index = self.base_index + height * self.step_size
        y_indices = torch.arange(start=self.base_index, end=y_end_index, step=self.step_size).type_as(smax)
        y_coords = torch.sum(torch.sum(smax, 3) * y_indices, 2)

        # For debugging (testing if it's actually like the argmax?)
        # argmax_x = argmax_x.view(batch_size, channels)
        # argmax_y = argmax_y.view(batch_size, channels)
        # print("X err in soft argmax: {err}".format(err=torch.mean(torch.abs(argmax_x - x_coords))))
        # print("Y err in soft argmax: {err}".format(err=torch.mean(torch.abs(argmax_y - y_coords))))

        # Put the x coords and y coords along with 1s of (shape (B,C)) into an output with shape (B,C,3)
        xs = torch.unsqueeze(x_coords, 2)
        ys = torch.unsqueeze(y_coords, 2)
        ones = torch.ones_like(xs)
        xys = torch.cat([xs, ys, ones], dim=2)
        xys = xys.view(batch_size, channels, 3)
        xys = xys.permute(0, 2, 1).contiguous()  # (batch, 3, njoint)

        maxv = maxv.view(batch_size, 1, channels)
        zero_xys = torch.zeros_like(xys)
        zero_xys[:, 2, :] = 1
        xys = torch.where(maxv > 0.01, xys, zero_xys)

        if hardmax:
            if out_smax:
                return xys_max, maxv.view(batch_size, channels), smax
            else:
                return xys_max, maxv.view(batch_size, channels)

        if out_smax:
            return xys, maxv.view(batch_size, channels), smax
        else:
            return xys, maxv.view(batch_size, channels)


def euclidean_to_homogeneous(points):
    """Converts euclidean points to homogeneous

    Args:
        points numpy array or torch tensor of shape (N, M): N euclidean points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M + 1): homogeneous points
    """
    if isinstance(points, np.ndarray):
        return np.hstack([points, np.ones((len(points), 1))])
    elif torch.is_tensor(points):
        return torch.cat([points, torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)], dim=1)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def homogeneous_to_euclidean(points):
    """Converts homogeneous points to euclidean

    Args:
        points numpy array or torch tensor of shape (N, M + 1): N homogeneous points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M): euclidean points
    """
    if isinstance(points, np.ndarray):
        return (points.T[:-1] / points.T[-1]).T
    elif torch.is_tensor(points):
        return (points.transpose(1, 0)[:-1] / points.transpose(1, 0)[-1]).transpose(1, 0)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def triangulate_point(proj_matrices, points, confidences=None):
    # modified from learnable triangulation/multiview
    """
    triangulate one joint in multi-views, in pytorch
    Args:
        proj_matrices: torch tensor in shape (n_cams, 3, 4), sequence of projection matricies (3x4)
        points: torch tensor in shape (N, 2), sequence of points' coordinates
        confidences: None or torch tensor of shape (N,), confidences of points [0.0, 1.0].
                                                    If None, all confidences are supposed to be 1.0

    Returns:
        point_3d numpy torch tensor of shape (3,): triangulated point
    """
    assert len(proj_matrices) == len(points)
    # ic(proj_matrices)
    n_views = len(proj_matrices)
    if confidences is None:
        confidences = torch.ones(n_views, dtype=torch.float32, device=points.device)
    # ic(confidences)
    A = proj_matrices[:, 2:3].expand(n_views, 2, 4) * points.view(n_views, 2, 1)
    A -= proj_matrices[:, :2]
    A *= confidences.view(-1, 1, 1)

    u, s, vh = torch.svd(A.view(-1, 4))
    point_3d_homo = -vh[:, 3]
    point_3d = homogeneous_to_euclidean(point_3d_homo.unsqueeze(0))[0]
    # compute triangulation residual
    # res_triang = torch.linalg.vector_norm(A @ point_3d_homo.unsqueeze(1), ord=1)
    res_triang = s[-1]
    return point_3d, res_triang


@HEADS.register_module()
class TriangulateHead(nn.Module):
    def __init__(self, num_cams=6, img_shape=[256, 256], heatmap_shape=[64, 64],
                 softmax_heatmap=True, loss_3d_sup=None, det_conf_thr=None,
                 train_cfg=None, test_cfg=None):
        super().__init__()
        self.num_cams = num_cams
        [self.h_img, self.w_img] = img_shape
        [self.h_map, self.w_map] = heatmap_shape
        self.det_conf_thr = det_conf_thr  # weather use the 2d detect confidence to mask the fail detection points
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        if loss_3d_sup is not None:  # and self.train_cfg.get('use_3d_sup')
            self.super_loss = build_loss(loss_3d_sup)

        self.smax = SoftArgmax2D(window_fn='Uniform', window_width=5 * 3, softmax_temp=0.05)  # window_width=5*hm_sigma

    def forward(self, heatmap, proj_matrices=None, confidences=None, reproject=True):
        """
        Args:
            heatmap: [bs*num_cams, num_joints, h_heatmap, w_heatmap]
            proj_matrices: [bs, num_cams, 3, 4]
            confidences: [bs*num_cams, num_joints]

        Returns:
            kp_3d: triangulation results, keypoint 3d coordinates, [bs, n_joints, 3]
            res_triang: triangulation residual, [bs, n_joints]
        """
        batch_size = int(heatmap.shape[0]/self.num_cams)
        n_joints = heatmap.shape[1]
        kp_2d_hm, maxv = self.smax(heatmap) # kp_2d_hm: [bs*num_cams, 3, num_joints], maxv: [bs*num_cams, num_joints]

        kp_2d_hm[:, -1, :] = maxv
        kp_2d_hm = kp_2d_hm.permute(0,2,1).reshape(batch_size , self.num_cams, n_joints, 3)

        kp_2d_croped = torch.zeros_like(kp_2d_hm, dtype=float)
        kp_2d_croped[:, :, :, 0] = kp_2d_hm[:, :, :, 0] * self.h_img / self.h_map
        kp_2d_croped[:, :, :, 1] = kp_2d_hm[:, :, :, 1] * self.w_img / self.w_map
        kp_2d_croped[:, :, :, 2] = kp_2d_hm[:, :, :, 2]

        kp_3d = torch.zeros(batch_size, n_joints, 3, dtype=torch.float32, device=kp_2d_croped.device)
        res_triang = torch.zeros(batch_size, n_joints, dtype=torch.float32, device=kp_2d_croped.device)

        confidences = confidences.view(batch_size, self.num_cams, *confidences.shape[1:])
        confidences = confidences / confidences.sum(dim=1, keepdim=True)  # [num_sample, num_cams, num_joints]
        confidences = confidences + 1e-5
        # ic(confidences[0])

        if self.det_conf_thr is not None:
            for batch_i in range(batch_size):
                for joint_i in range(n_joints):
                    cams_detected = kp_2d_croped[batch_i, :, joint_i, 2] > self.det_conf_thr
                    cam_idx = torch.where(cams_detected)[0]
                    point = kp_2d_croped[batch_i, cam_idx, joint_i, :2]  # a joint in all views
                    confidence = confidences[batch_i, cam_idx, joint_i]
                    if torch.sum(cams_detected) < 2:
                        continue
                    kp_3d[batch_i, joint_i], res_triang[batch_i, joint_i] = \
                        triangulate_point(proj_matrices[batch_i, cam_idx], point, confidence)
        else:
            for batch_i in range(batch_size):
                for joint_i in range(n_joints):
                    points = kp_2d_croped[batch_i, :, joint_i, :2]  # a joint in all views
                    confidence = confidences[batch_i, :, joint_i]
                    kp_3d[batch_i, joint_i], res_triang[batch_i, joint_i] = \
                        triangulate_point(proj_matrices[batch_i], points, confidence)

        if reproject:
            reproject_kp_2d = self.reproject(kp_3d, proj_matrices)
        else:
            reproject_kp_2d = None

        return kp_3d, res_triang, kp_2d_croped, reproject_kp_2d, kp_2d_hm

    def reproject(self, kp_3d, proj_matrices):
        """
        Args:
            kp_3d: np.array, [bs, n_joints, 3]
            proj_matrices: [bs, num_cams, 3, 4]
        Returns:
            reproject_kp_2d: [bs, num_cams, num_joints, 2]
        """
        # if kp_3d.shape[-1] == 3:
        #     kp_3d_temp = np.concatenate([kp_3d, torch.ones([*kp_3d.shape[:-1], 1])], dim=-1)
        # pseudo_kp_2d = np.einsum('bcdk, bjk -> bcjd', proj_matrices, kp_3d_temp)
        # pseudo_kp_2d = pseudo_kp_2d/(np.expand_dims(pseudo_kp_2d[..., -1], -1))
        # pseudo_kp_2d = pseudo_kp_2d[..., :-1]
        # return pseudo_kp_2d
        kp_3d_temp = kp_3d.detach().clone()
        if kp_3d_temp.shape[-1] == 3:
            kp_3d_temp = torch.cat(
                [kp_3d_temp,
                 torch.ones([*kp_3d_temp.shape[:-1], 1], dtype=torch.float64, device=kp_3d_temp.device)],
                dim=-1)

        pseudo_kp_2d = torch.einsum('bcdk, bjk -> bcjd', proj_matrices, kp_3d_temp)
        pseudo_kp_2d = pseudo_kp_2d / (pseudo_kp_2d[..., -1].unsqueeze(-1))
        pseudo_kp_2d = pseudo_kp_2d[..., :-1]
        return pseudo_kp_2d

    def get_sup_loss(self, output, target, target_visible=None):
        """Calculate supervised 3d keypoint regressive loss.

        Args:
            output (torch.Tensor[bs, num_joints, 3]): Output 3d keypoint coordinates.
            target (torch.Tensor[bs, num_joints, 3]): Target 3d keypoint coordinates.
            target_weight (torch.Tensor[bs, num_joints, 1]):
                Weights across different joint types.
        """

        losses = dict()

        assert not isinstance(self.super_loss, nn.Sequential)
        # assert target.dim() == 3 and target_weight.dim() == 3
        # print(output.shape, target.shape, target_visible.shape)
        # print(output)
        # print(target)
        # print(target_visible > 0)
        # print(output[target_visible > 0].shape)
        output = output[target_visible > 0].double()
        target = target[target_visible > 0].double()

        losses['sup_3d_loss'] = self.super_loss(output, target)
        return losses

    def get_unSup_loss(self, res_triang):
            """
            Calculate the triangulation residual loss, unsupervised 3d loss
            Args:
                res_triang: [bs, ??]
            Returns:

            """
            losses = dict()
            losses['unSup_3d_loss'] = torch.mean(res_triang)

            return losses