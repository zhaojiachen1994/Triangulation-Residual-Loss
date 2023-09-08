import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import constant_init, normal_init
from mmpose.models.builder import HEADS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, normal_init
from torch.autograd import Function



class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor, coeff= 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.coeff, None

class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

class WarmStartGradientReverseLayer(nn.Module):
    """Gradient Reverse Layer :math:`\mathcal{R}(x)` with warm start

        The forward and backward behaviours are:

        .. math::
            \mathcal{R}(x) = x,

            \dfrac{ d\mathcal{R}} {dx} = - \lambda I.

        :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:

        .. math::
            \lambda = \dfrac{2(hi-lo)}{1+\exp(- α \dfrac{i}{N})} - (hi-lo) + lo

        where :math:`i` is the iteration step.

        Args:
            alpha (float, optional): :math:`α`. Default: 1.0
            lo (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            hi (float, optional): Final value of :math:`\lambda`. Default: 1.0
            max_iters (int, optional): :math:`N`. Default: 1000
            auto_step (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha=1.0, lo=0.0, hi=1.,
                 max_iters=1000., auto_step=False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1

def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct

@HEADS.register_module()
class DomainDiscriminator(nn.Module):
    r"""Domain discriminator model from
        `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_

        Distinguish whether the input features come from the source domain or the target domain.
        The source domain label is 1 and the target domain label is 0.

        Args:
            in_feature (int): dimension of the input feature
            hidden_size (int): dimension of the hidden features
            batch_norm (bool): whether use :class:`~torch.nn.BatchNorm1d`.
                Use :class:`~torch.nn.Dropout` if ``batch_norm`` is False. Default: True.

        Shape:
            - Inputs: (minibatch, `in_feature`)
            - Outputs: :math:`(minibatch, 1)`
        """

    def __init__(self, in_channels: int, hidden_size: int,
                 in_index=0,
                 first_layer = 'conv', # pool
                 input_transform=None,
                 sigmoid=True,
                 reduction='mean',
                 use_weight=False,
                 grl=None,

                 # train_cfg=None,
                 # test_cfg=None
                 ):
        super().__init__()
        # self.train_cfg = {} if train_cfg is None else train_cfg
        # self.test_cfg = {} if test_cfg is None else test_cfg
        self.in_index = in_index
        self.input_transform = input_transform
        self.sigmoid = sigmoid
        self.reduction = reduction
        self.use_weight = use_weight

        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=100,
                                                 auto_step=True) if grl is None else grl
        if sigmoid:
            final_layer = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            final_layer = nn.Linear(hidden_size, 2)

        if first_layer == 'conv':
            first_layer = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1)
            neck_channels = 4096

        elif first_layer == 'pool':
            first_layer = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            neck_channels = 48

        layers = [
            first_layer,
            nn.Flatten(),
            nn.Linear(neck_channels, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            final_layer
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, num_source_img=1, num_target_img=1):
        """
        Args:
            x: feature maps from backbone, combining source and target samples
        Returns:
            d_s, d_t: the domain discriminator predicted labels for source/target domain
            d_label_s, d_label_t: ground truth domain labels
        """
        x = self._transform_inputs(x)
        x = self.grl(x) # WarmStartGradientReverseLayer
        d = self.layers(x)
        if self.sigmoid:
            d_s = d[:num_source_img]    # the
            d_t = d[-num_target_img:]
            d_label_s = torch.ones((num_source_img, 1)).to(d_s.device)
            d_label_t = torch.zeros((num_target_img, 1)).to(d_s.device)
            # ic(d_label_s, d_label_t)


        return d_s, d_t, d_label_s, d_label_t

    def get_loss(self, d_s, d_t, d_label_s, d_label_t):
        losses = dict()
        if self.use_weight:
            w_s = torch.ones_like(d_s).to(d_s.device)
            w_t = (torch.ones_like(d_t)*d_s.size(0)/d_t.size(0)).to(d_t.device)
            losses['dd_loss'] = 0.1 * 0.5 * (
                    F.binary_cross_entropy(d_s, d_label_s, weight=w_s, reduction=self.reduction) +
                    F.binary_cross_entropy(d_t, d_label_t, weight=w_t, reduction=self.reduction)
            )
        else:
            losses['dd_loss'] = 0.1 * 0.5 * (
                    F.binary_cross_entropy(d_s, d_label_s, reduction=self.reduction) +
                    F.binary_cross_entropy(d_t, d_label_t, reduction=self.reduction))
        return losses

    def get_accuracy(self, d_s, d_t, d_label_s, d_label_t):
        accuracy = dict()
        accuracy['acc_dd'] = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))
        return accuracy

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        from topdown_heatmap_simple_head.py

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(inputs, list):
            return inputs

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def init_weights(self):
        for _, m in self.layers.named_modules():
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm1d):
                constant_init(m, 1)
