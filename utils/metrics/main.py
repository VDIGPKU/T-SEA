import shutil

from . import draw_mAP
import os

class Args:
    def __init__(self, path='./', lab_path='det-results', gt_path='ground-truth', res_prefix='',
                 no_animation=False, no_plot=False, quiet=False, ignore=None, set_class_iou=None):
        self.path = path
        self.lab_path = lab_path
        self.gt_path = gt_path
        self.res_prefix = res_prefix
        self.no_animation = no_animation
        self.no_plot = no_plot
        self.quiet = quiet
        self.ignore = ignore
        self.set_class_iou = set_class_iou


def compute_mAP(path='./', lab_path='det-results', gt_path='ground-truth', res_prefix='',
         no_animation=False, no_plot=False, quiet=False, ignore=None, set_class_iou=None):
    args = Args(path, lab_path, gt_path, res_prefix, no_animation, no_plot, quiet, ignore, set_class_iou)
    ap_dictionary, mAP = draw_mAP(args)
    return mAP


def compute_acc():
    pass
