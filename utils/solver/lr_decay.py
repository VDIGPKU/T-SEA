import numpy as np


def warm_up_cosine_decay(lr_max=0.6, lr_min=0.05, warmup_epochs=0, total_epochs=1000):
    # learning_rate = 0.5 * learning_rate_base * (
    #         1 + np.cos(np.pi * (global_step - warmup_epochs - hold_base_rate_steps) / float(
    #     total_epochs - warmup_epochs - hold_base_rate_steps)
    #                    )
    # )
    pass


def cosine_decay(cur_epochs, lr_max=0.5, lr_min=0.05, total_epochs=1000):
    '''

    :param cur_epochs:
    :param lr_max:
    :param lr_min:
    :param total_epochs:
    :return:

    attack_confs_thresh = cosine_decay(epoch-1, lr_max=cfg.DETECTOR.CONF_THRESH, lr_min=0.05)
    '''
    pos = np.pi * (cur_epochs/total_epochs) / 2
    learning_rate = lr_min + 1/2 * (lr_max - lr_min) * (1 + np.cos(pos))
    return learning_rate
