import math
from torch import nn
from torch.autograd import Function
import torch
import omg_cuda


class SDFLossFunction(Function):
    @staticmethod
    def forward(
        ctx,
        pose_init,
        sdf_grids,
        sdf_limits,
        points,
        epsilons,
        padding_scales,
        clearances,
        disables,
    ):
        outputs = omg_cuda.sdf_loss_forward(
            pose_init,
            sdf_grids,
            sdf_limits,
            points,
            epsilons,
            padding_scales,
            clearances,
            disables,
        )
        potentials = outputs[0]
        potential_grads = outputs[1]
        collides = outputs[2]

        return potentials, potential_grads, collides

    @staticmethod
    def backward(ctx, top_diff, top_diff_grad):
        return None, None, None, None, None, None, None


class SDFLoss(nn.Module):
    def __init__(self):
        super(SDFLoss, self).__init__()

    def forward(
        self,
        pose_init,
        sdf_grids,
        sdf_limits,
        points,
        epsilons,
        padding_scales,
        clearances,
        disables,
    ):
        return SDFLossFunction.apply(
            pose_init,
            sdf_grids,
            sdf_limits,
            points,
            epsilons,
            padding_scales,
            clearances,
            disables,
        )
