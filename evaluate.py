import argparse
import os

import numpy as np
import torch
import shutil
from tqdm import tqdm

from attack.attacker import UniversalAttacker
from utils.preprocesser.gen_det_labels import Utils
from utils.parser import ConfigParser, logger_cfg
from utils.metrics.main import compute_mAP
from scripts.dict import MAP_PATHS

import warnings
warnings.filterwarnings('ignore')
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# from utils.det_utils import plot_boxes_cv2
label_postfix = '-rescale-labels'

def path_remove(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path) # a dir
        except:
            os.remove(path) # a symbolic link

def dir_check(save_path, child_paths, rebuild=False):
    # if the target path exists, it will be deleted (for empty dirt) and rebuild-up
    def buid(path, rebuild):
        if rebuild:
            path_remove(path)
        try:
            os.makedirs(path, exist_ok=True)
        except:
            pass
    buid(save_path, rebuild=rebuild)
    for child_path in child_paths:
        child_path = child_path.lower()
        tmp_path = os.path.join(save_path, child_path)
        for path in MAP_PATHS.values():
            ipath = os.path.join(tmp_path, path)
            buid(ipath, rebuild)


class UniversalPatchEvaluator(UniversalAttacker):
    def __init__(self, cfg, patch_path=None,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 ):
        super().__init__(cfg, device)
        self.cfg = cfg
        self.device = device
        if patch_path is not None:
            self.patch_obj.read(patch_path)

    def read_patch_from_memory(self, patch):
        self.patch_obj.update_(patch)


def get_save(args):
    def get_prefix(path):
        if os.sep in path:
            path = path.split(os.sep)[-1]
        return path.split('.')[0]
    prefix = get_prefix(args.patch)
    args.save = os.path.join(args.save, prefix)
    return args


def ignore_class(args, cfg):
    # Be careful of the so-called 'attack_list' and 'eva_class' in the evaluate.py
    # For higher reusability of the codes, these variable names may be confusing
    # In this file, the 'attack_list' is loaded from the config file which has been used for training
    # (cuz we don't bother to create a new config file for evaluations)
    # Thus the 'attack_list' refers to the original attacked classes when training the patch
    # while the 'eva_list' denotes the class list to be evaluated, which are to attack in the evaluation
    # (When the eva classes are different from the original attack classes,
    # it is mainly for the partial attack in evaluating unseen-class/cross-class performance)
    args.eva_class_list = cfg.rectify_class_list(args.eva_class, dtype='str')
    # print('Eva(Attack) classes from evaluation: ', cfg.show_class_index(args.eva_class_list))
    # print('Eva classes names from evaluation: ', args.eva_class_list)

    args.ignore_class = list(set(cfg.all_class_names).difference(set(args.eva_class_list)))
    if len(args.ignore_class) == 0: args.ignore_class = None
    return args


def generate_labels(evaluator, cfg, args, save_label=False):
    from utils.loader import dataLoader
    dir_check(args.save, cfg.DETECTOR.NAME, rebuild=False)
    utils = Utils(cfg)
    batch_size = 1

    data_loader = dataLoader(args.data_root, input_size=cfg.DETECTOR.INPUT_SIZE,
                             batch_size=batch_size, is_augment=False, pin_memory=True)
    img_names = [os.path.join(args.data_root, i) for i in os.listdir(args.data_root)]
    save_path = args.save
    accs_total = {}
    for detector in evaluator.detectors: accs_total[detector.name] = []
    # print(evaluator.detectors)
    for index, img_batch in enumerate(tqdm(data_loader, total=len(data_loader))):
        names = img_names[index:index + batch_size]
        img_name = names[0].split('/')[-1]
        for detector in evaluator.detectors:
            # make sure every detector detect in a new batch of img tensors (avoid of the inplace)
            img_tensor_batch = img_batch.to(evaluator.device)
            tmp_path = os.path.join(save_path, detector.name)
            all_preds = evaluator.detect_bbox(img_tensor_batch, detectors=[detector])
            evaluator.get_patch_pos_batch(all_preds)
            if save_label:
                # for saving the original detection info
                fp = os.path.join(tmp_path, MAP_PATHS['det-lab'])
                utils.save_label(all_preds[0], fp, img_name, save_conf=False, rescale=True)

            if hasattr(args, 'test_origin') and args.test_origin:
                fp = os.path.join(tmp_path, MAP_PATHS['det-res'])
                utils.save_label(all_preds[0], fp, img_name, save_conf=True, rescale=True)

            target_nums_clean = evaluator.get_patch_pos_batch(all_preds)[0]
            adv_img_tensor = evaluator.uap_apply(img_tensor_batch)

            preds = detector(adv_img_tensor)['bbox_array']
            if hasattr(args, 'save_imgs') and args.save_imgs:
                # for saving the attacked imgs
                ipath = os.path.join(tmp_path, 'imgs')
                evaluator.plot_boxes(adv_img_tensor[0], preds[0], save_path=ipath, save_name=img_name)

            # for saving the attacked detection info
            lpath = os.path.join(tmp_path, MAP_PATHS['attack-lab'])
            utils.save_label(preds[0], lpath, img_name, rescale=True)


def eval_init(args, cfg, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    # preprocessing the cfg
    args = ignore_class(args, cfg)
    evaluator = UniversalPatchEvaluator(cfg, args.patch, device)

    cfg = cfg_save_modify(cfg)
    return args, cfg, evaluator


def cfg_save_modify(cfg):
    cfg.DETECTOR.PERTURB.GATE = None
    cfg.DATA.AUGMENT = ''
    return cfg


def eval_patch(args, cfg):
    args, cfg, evaluator = eval_init(args, cfg)
    logger_cfg(args, 'Evaluating')
    evaluator.attack_list = cfg.show_class_index(args.eva_class_list)

    if not args.gen_no_label:
        generate_labels(evaluator, cfg, args)
    save_path = args.save

    det_mAPs = {}; gt_mAPs = {}; ori_mAPs = {}
    quiet = args.quiet if hasattr(args, 'quiet') else False
    # to compute mAP
    for detector in evaluator.detectors:
        # tmp_path = os.path.join(cfg.ATTACK_SAVE_PATH, detector.name)
        path = os.path.join(save_path, detector.name)

        # link the path of the detection labels
        det_path = os.path.join(path, MAP_PATHS['det-lab'])
        path_remove(det_path)

        # cmd = 'ln -s ' + os.path.join(args.label_path, detector.name+'-labels') + ' ' + det_path
        source = os.path.join(args.label_path, detector.name + label_postfix)
        cmd = ' '.join(['ln -s ', source, det_path])
        # print(cmd)
        os.system(cmd)

        # (det-results)take clear detection results as GT label: attack results as detections
        # print('ground truth     :', os.path.join(path, MAP_PATHS['det-lab']))
        det_mAP = compute_mAP(path=path, ignore=args.ignore_class, lab_path=MAP_PATHS['attack-lab'],
                                gt_path=MAP_PATHS['det-lab'], res_prefix='det', quiet=quiet)
        det_mAPs[detector.name] = round(det_mAP*100, 2)
        # shutil.rmtree(os.path.join(path, MAP_PATHS['attack-lab']))

        if hasattr(args, 'test_gt') and args.test_gt:
            # link the path of the GT labels
            gt_target = os.path.join(path, 'ground-truth')
            gt_source = os.path.join(args.label_path, MAP_PATHS['ground-truth'] + label_postfix)
            path_remove(gt_target)
            cmd = ' '.join(['ln -s ', gt_source, gt_target])
            print(cmd)
            os.system(cmd)
            # (gt-results)take original labels as GT label(default): attack results as detections
            # print('ground truth     :', MAP_PATHS['ground-truth'])
            gt_mAP = compute_mAP(path=path, ignore=args.ignore_class, lab_path=MAP_PATHS['attack-lab'],
                                    gt_path=MAP_PATHS['ground-truth'], res_prefix='gt', quiet=quiet)
            gt_mAPs[detector.name] = round(gt_mAP*100, 2)

        if hasattr(args, 'test_origin') and args.test_origin:
            rp = 'ori'
            # (ori-results)take original labels as path['ground-truth'] label(default): clear detection res as detections
            ori_mAP = compute_mAP(path=path, ignore=args.ignore_class, lab_path=MAP_PATHS['det-res'],
                                gt_path=MAP_PATHS['ground-truth'], res_prefix=rp, quiet=quiet)
            ori_mAPs[rp][detector.name] = round(ori_mAP*100, 2)

    return det_mAPs, gt_mAPs, ori_mAPs


if __name__ == '__main__':
    from utils.parser import dict2txt, merge_dict_by_key
    # To test attack performance with reference to detection labels.

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, default=None, help="Adversarial patch path.")
    parser.add_argument('-cfg', '--cfg', type=str, default=None, help="A relative file path of teh .yaml cfg file.")
    parser.add_argument('-s', '--save', type=str, default=os.path.join(PROJECT_DIR, '/data/inria/'), help="Directory to save evaluation results.")
    parser.add_argument('-lp', '--label_path', type=str, default=os.path.join(PROJECT_DIR, '/INRIAPerson/Test/labels'), help='ground truth & detector predicted labels dir')
    parser.add_argument('-dr', '--data_root', type=str, default=os.path.join(PROJECT_DIR, '/data/INRIAPerson/Test/pos'))
    parser.add_argument('-to', '--test_origin', action='store_true', help="To test detector performance in clean samples.")
    parser.add_argument('-tg', '--test_gt', action='store_true', help="To test attack performance with reference to ground truth labels(Annotation).")
    parser.add_argument('-ul', '--stimulate_uint8_loss', action='store_true', help="To stimulate uint8 loss from float preprocesser format.")
    parser.add_argument('-i', '--save_imgs', help='to save attacked imgs', action='store_true')
    parser.add_argument('-ng', '--gen_no_label', action='store_true', help="Won't generate any detection labels of adversarial samples if set True.")
    parser.add_argument('-e', '--eva_class', type=str, default='0', help="The class to attack. '-1': all classes, '-2': attack seen classes(ATTACK_CLASS in cfg file), '-3': attack unseen classes(all_class - ATTACK_CLASS); or custom '0, 2:5, 10'.")
    parser.add_argument('-q', '--quiet', action='store_true', help='logger none if set true')
    args = parser.parse_args()
    cfg = ConfigParser(args.cfg)

    args = get_save(args)
    # args, evaluator = eval_init(args, cfg)
    det_mAPs, gt_mAPs, ori_mAPs = eval_patch(args, cfg)

    det_mAP_file = os.path.join(args.save, 'det-mAP.txt')
    if not os.path.exists(det_mAP_file):
        with open(det_mAP_file, 'a') as f:
            f.write('              scale : ' + str(cfg.ATTACKER.PATCH.SCALE) + '\n')
            f.write('--------------------------\n')

    det_dict = det_mAPs
    dict2txt(det_dict, det_mAP_file)
    dict2txt(gt_mAPs, os.path.join(args.save, 'gt-mAP.txt'))
    if not args.quiet:
        print("det dict      mAP :", det_dict)

