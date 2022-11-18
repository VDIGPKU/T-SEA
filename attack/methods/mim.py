from .base import BaseAttacker

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import copy
from tqdm import tqdm
import cv2


class LinfMIMAttack(BaseAttacker):
    """MI-FGSM attack (arxiv: https://arxiv.org/pdf/1710.06081.pdf)"""
    def __init__(self, loss_func, cfg, device, detector_attacker, norm='L_infty', momentum=0.9):
        super().__init__(loss_func, norm, cfg, device, detector_attacker)
        self.momentum = momentum
        self.grad = None

        # self.param_groups = [{'lr': cfg.STEP_LR}]

    @property
    def step_lr(self):
        return self.param_groups[0]['lr']

    def patch_update(self, **kwargs):
        now_grad = self.patch_obj.patch.grad
        if self.grad is None:
            self.grad = now_grad
        else:
            self.grad = self.grad * self.momentum + now_grad / torch.norm(now_grad, p=1)
        update = self.step_lr * self.grad.sign()

        if "descend" in self.cfg.LOSS_FUNC:
            update *= -1
        patch_tmp = self.patch_obj.patch + update
        patch_tmp = torch.clamp(patch_tmp, min=self.min_epsilon, max=self.max_epsilon)
        self.patch_obj.update_(patch_tmp)
