import copy
import sys

from .base import BaseAttacker

import torch
import torch.distributed as dist
import numpy as np

num_iter = 0
update_pre = 0
# patch_tmp = None

import sys
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.append(PROJECT_ROOT)
from utils import FormatConverter


class LinfPGDAttack(BaseAttacker):
    """PGD attacks (arxiv: https://arxiv.org/pdf/1706.06083.pdf)"""
    def __init__(self, loss_func, cfg, device, detector_attacker, norm='L_infty'):
        super().__init__(loss_func, norm, cfg, device, detector_attacker)
        self.step_lr = cfg.STEP_LR

    def patch_update(self, **kwargs):
        update = self.step_lr * self.patch_obj.patch.grad.sign()
        if "descend" in self.cfg.LOSS_FUNC:
            update *= -1

        patch_tmp = self.patch_obj.patch + update
        torch.clamp_(patch_tmp.data, min=self.min_epsilon, max=self.max_epsilon)
        self.patch_obj.update_(patch_tmp)
        return patch_tmp

    def parallel_non_targeted_attack(self, ori_tensor_batch, detector_attacker, detector):
        adv_tensor_batch, patch_tmp = detector_attacker.uap_apply(ori_tensor_batch)
        loss = []
        for iter in range(self.iter_step):
            preds, confs = detector(adv_tensor_batch)
            disappear_loss = self.attack_loss(confs)
            loss.append(float(disappear_loss))
            detector.zero_grad()
            disappear_loss.backward()
            self.patch_update(patch_tmp=patch_tmp)
            adv_tensor_batch, _ = detector_attacker.uap_apply(ori_tensor_batch, universal_patch=patch_tmp)

        return patch_tmp
            