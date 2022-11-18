from bisect import bisect_right
import torch
from torch import optim


class ALRS():
    """ALRS is a scheduler without warmup, a variant of warmupALRS.
    ALRS decays the learning rate when
    """
    def __init__(self, optimizer, loss_threshold=1e-4, loss_ratio_threshold=1e-4, decay_rate=0.97, patience=10):
        self.optimizer = optimizer
        self.loss_threshold = loss_threshold
        self.decay_rate = decay_rate
        self.loss_ratio_threshold = loss_ratio_threshold
        self.last_loss = 999
        self.total_epoch_loss = 0
        self.patience = patience

    def update_lr(self, loss):
        delta = self.last_loss - loss
        if delta < self.loss_threshold and delta / self.last_loss < self.loss_ratio_threshold:
            for ind, group in enumerate(self.optimizer.param_groups):
                self.optimizer.param_groups[ind]['lr'] *= self.decay_rate
                now_lr = group['lr']
                print(f'now lr = {now_lr}')

    def step(self, **kargs):
        epoch = kargs['epoch']
        loss = kargs['ep_loss']
        if epoch % self.patience != 0:
            self.total_epoch_loss += loss
        else:
            loss = self.total_epoch_loss / self.patience
            self.update_lr(loss)
            self.last_loss = loss
            self.total_epoch_loss = 0


class warmupALRS(ALRS):
    """reference:Bootstrap Generalization Ability from Loss Landscape Perspective"""

    def __init__(self, optimizer, warmup_epoch=50, loss_threshold=1e-4, loss_ratio_threshold=1e-4, decay_rate=0.97):
        super().__init__(optimizer, loss_threshold, loss_ratio_threshold, decay_rate)
        self.warmup_rate = 1 / 3
        self.warmup_epoch = warmup_epoch
        self.start_lr = optimizer.param_groups[0]["lr"]
        self.warmup_lr = self.start_lr * (1 - self.warmup_rate)
        self.update_lr(lambda x: x * self.warmup_rate)

    def update_lr(self, update_fn):
        for ind, group in enumerate(self.optimizer.param_groups):
            self.optimizer.param_groups[ind]['lr'] = update_fn(group['lr'])
            now_lr = group['lr']
            print(f'now lr = {now_lr}')

    def step(self, **kwargs):
        loss = kwargs['ep_loss']
        epoch = kwargs['epoch']
        if epoch < self.warmup_epoch:
            self.update_lr(lambda x: -(self.warmup_epoch-epoch)*self.warmup_lr / self.warmup_epoch + self.start_lr)
        elif epoch % self.patience != 0:
            self.total_epoch_loss += loss
        else:
            loss = self.total_epoch_loss / self.patience
            delta = self.last_loss - loss
            self.last_loss = loss
            if delta < self.loss_threshold and delta / self.last_loss < self.loss_ratio_threshold:
                self.update_lr(lambda x: x*self.decay_rate)


class ALRS_LowerTV(ALRS):
    """
    A variant of the standard ALRS.
    This is just for observational scheduler comparison of the optimization to the Plateau_LR
        employed in the current baseline <Fooling automated surveillance cameras: adversarial patches to attack person detection>.
    The difference is that we fine-tune the hyper-params decay_rate
        to force the learning rate down to 0.1 so that the TV Loss will converges to the same level.
    """

    def __init__(self, optimizer, loss_threshold=1e-4, loss_ratio_threshold=1e-4, decay_rate=0.94):
        super().__init__(optimizer, loss_threshold, loss_ratio_threshold, decay_rate)


class CosineLR():
    def __init__(self, optimizer, total_epoch=1000):
        self.optim = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch)

    def step(self, **kwargs):
        self.optim.step()


class ExponentialLR():
    def __init__(self, optimizer):
        self.optim = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999, update_step=5)

    def step(self, **kwargs):
        self.optim.step()


class PlateauLR():
    def __init__(self, optimizer, type='min', patience=100):
        self.optim = optim.lr_scheduler.ReduceLROnPlateau(optimizer, type, patience=patience)

    def step(self, **kargs):
        ep_loss = kargs['ep_loss']
        self.optim.step(ep_loss)
