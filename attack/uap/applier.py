"""

This is not used since tons of tensors takes huge GPU memory
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .transformer import PatchTransformer


class PatchRandomApplier(nn.Module):
    # apply patch
    def __init__(self, device: torch.device, cfg_patch: object):
        """

        :param rotate_angle: random rotate angle range from [-rotate_angle, rotate_angle]
        :param rand_loc_rate: random shift rate (of patch) range
        :param scale_rate: patch size / bbox size
        """
        super().__init__()
        self.cfg = cfg_patch
        self.patch_transformer = PatchTransformer(device, cfg_patch).to(device)
        self.device = device

    def forward(self, img_batch: torch.Tensor, adv_patch: torch.Tensor, bboxes_batch: torch.Tensor):
        """ This func to process the bboxes list of mini-batch into uniform torch.tensor and
        apply the patch into the img batch. Every patch stickers will be randomly transformed
        by given transform range before being attached.

        :param img_batch: image batch
        :param adv_patch: the adversarial patch
        :param bboxes_batch: bbox [batch_size, [N*6]]
        :return:
        """
        # print(img_batch.size, adv_patch.size)
        gates = patch_aug_gates(self.cfg.TRANSFORM)
        patch_ori_size = adv_patch.size(-1)
        batch_size = img_batch.size(0)
        pad_size = (img_batch.size(-1) - adv_patch.size(-1)) / 2
        padding = nn.ConstantPad2d((int(pad_size + 0.5), int(pad_size), int(pad_size + 0.5), int(pad_size)), 0)  # (LRTB)

        lab_len = bboxes_batch.size(1)
        # --------------Median pool degradation & Random jitter---------------------
        adv_batch = adv_patch.unsqueeze(0)
        if gates['median_pool']:
            adv_batch = self.patch_transformer.median_pooler(adv_batch[0])
        adv_batch = adv_batch.expand(batch_size, lab_len, -1, -1, -1) # [batch_size, lab_len, 3, N, N]
        if gates['jitter']:
            adv_batch = self.patch_transformer.random_jitter(adv_batch)
        adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)
        if gates['cutout']:
            adv_batch = self.patch_transformer.cutout(adv_batch)
        adv_batch = padding(adv_batch)

        # transform by gates
        adv_batch_t = self.patch_transformer(adv_batch, bboxes_batch, patch_ori_size,
                                             rand_rotate_gate=gates['rotate'],
                                             rand_shift_gate=gates['shift'])

        adv_img_batch = PatchApplier.forward(img_batch, adv_batch_t)
        # print('Patch apply out: ', adv_img_batch.shape)
        return adv_img_batch


class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """
    @staticmethod
    def forward(img_batch, adv_batch):
        advs = torch.unbind(adv_batch, 1)
        for adv in advs:
            img_batch = torch.where((adv == 0), img_batch, adv)
        return img_batch



def patch_aug_gates(aug_list):
    gates = {'jitter': False, 'median_pool': False, 'rotate': False, 'shift': False, 'cutout': False}
    for aug in aug_list:
        gates[aug] = True
    return gates