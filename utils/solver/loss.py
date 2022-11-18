import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


# def disappear_loss(cls_score):
#     mseloss = torch.nn.MSELoss()
#     loss = 1. - mseloss(cls_score[cls_score>=0.3], torch.zeros(cls_score[cls_score>=0.3].shape).cuda())
#     return loss


class TVLoss(nn.Module):
    @staticmethod
    def forward(inputs, t_mask=None):
        logo = inputs.permute(0, 2, 3, 1)
        vert_diff = logo[:, 1:] - logo[:, :-1]
        hor_diff = logo[:, :, 1:] - logo[:, :, :-1]
        vert_diff_sq = vert_diff ** 2
        hor_diff_sq = hor_diff ** 2
        vert_pad = F.pad(vert_diff_sq, (0, 0, 0, 0, 1, 0, 0, 0))
        hor_pad = F.pad(hor_diff_sq, (0, 0, 1, 0, 0, 0, 0, 0))
        tv_sum = vert_pad + hor_pad
        tv = (tv_sum + 1e-5).sqrt()
        tv_final_sum = torch.sum(tv)
        tv_loss = tv_final_sum
        return tv_loss

    @staticmethod
    def smooth(adv_patch):
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), 0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), 0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv = tvcomp1 + tvcomp2
        return tv / torch.numel(adv_patch)


def obj_tv_loss(**kwargs):
    confs = kwargs['confs']; patch = kwargs['patch']
    tv_loss = TVLoss.smooth(patch)
    obj_loss = torch.mean(confs)
    loss = {'tv_loss': tv_loss, 'obj_loss': obj_loss}
    return loss


def descend_mse_loss(**kwargs):
    # print(confs.shape)
    confs = kwargs['confs']
    target = torch.cuda.FloatTensor(confs.shape).fill_(0)
    return torch.nn.MSELoss()(confs, target)


def ascend_mse_loss(**kwargs):
    # print(confs.shape)
    confs = kwargs['confs']
    target = torch.cuda.FloatTensor(confs.shape).fill_(1)
    return torch.nn.MSELoss()(confs, target)


def attack_loss(det_score, det_labels, cls_scores, class_id, patch):
    disappear_loss = DisappearLoss()
    class_error_loss = ClassErrorLoss()
    tv_loss = TVLoss()
    return disappear_loss(det_score, det_labels, class_id) + class_error_loss(cls_scores, class_id) - 10 * tv_loss(patch)

class DisappearLoss(nn.Module):

    def __init__(self):
        super(DisappearLoss, self).__init__()

    def forward(self, det_bboxes, det_labels, class_id):
        mseloss = torch.nn.MSELoss()
        if class_id != -1 and class_id in list(det_labels[0].cpu().numpy()):
            # 指定类别攻击
            select_id = np.where(det_labels[0].cpu().numpy() == class_id)
            loss = 1. - mseloss(det_bboxes[select_id], torch.zeros(det_bboxes[select_id].shape).cuda())
        else:
            # 无指定类别攻击
            loss = 1. - mseloss(det_bboxes[det_bboxes >= 0.01],
                                torch.zeros(det_bboxes[det_bboxes >= 0.01].shape).cuda())
        return loss


class ClassErrorLoss(nn.Module):

    def __init__(self):
        super(ClassErrorLoss, self).__init__()

    def forward(self, cls_scores, class_id):
        mseloss = torch.nn.MSELoss()
        if class_id != -1:
            # 指定类别攻击
            loss = 1. - mseloss(cls_scores[:, class_id], -torch.ones(cls_scores[:, class_id].shape).cuda())
        else:
            # 无指定类别攻击
            loss = 1. - mseloss(cls_scores, -torch.ones(cls_scores.shape).cuda())
        return loss
