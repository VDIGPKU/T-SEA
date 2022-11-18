from .base import BaseAttacker

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import cv2
import copy
from tqdm import tqdm


class LinfBIMAttack(BaseAttacker):
    """BIM attack (arxiv: https://arxiv.org/pdf/1607.02533.pdf)
    """
    def __init__(self, loss_func, cfg, device, detector_attacker, norm='L_infty'):
        super().__init__(loss_func, norm, cfg, device, detector_attacker)
        self.epsilon = 0.1

        # self.param_groups = [{'lr': cfg.STEP_LR}]

    @property
    def step_lr(self):
        return self.param_groups[0]['lr']

    def patch_update(self, **kwargs):
        grad = self.patch_obj.patch.grad
        update = self.step_lr * grad.sign()

        if "descend" in self.cfg.LOSS_FUNC:
            update *= -1
        update = torch.clamp(update, min=-self.epsilon, max=self.epsilon)
        patch_tmp = self.patch_obj.patch + update
        patch_tmp = torch.clamp(patch_tmp, min=self.min_epsilon, max=self.max_epsilon)
        self.patch_obj.update_(patch_tmp)