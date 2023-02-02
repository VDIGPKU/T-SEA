import argparse
import os

import numpy as np
import torch
import shutil
from tqdm import tqdm

from attack.attacker import UniversalAttacker
from utils.preprocesser.gen_det_labels import Utils
from utils.parser import ConfigParser, logger_cfg, ignore_class
from utils.metrics.main import compute_mAP
from scripts.dict import MAP_PATHS
from utils.utils import dir_check, path_remove

import warnings
warnings.filterwarnings('ignore')
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# from utils.det_utils import plot_boxes_cv2
label_postfix = '-rescale-labels'


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
    # if os.path.exists(args.save):
    #     shutil.rmtree(args.save)
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
                utils.save_label((all_preds[0]).clone(), fp, img_name, save_conf=False, rescale=True)

            if hasattr(args, 'test_origin') and args.test_origin:
                fp = os.path.join(tmp_path, MAP_PATHS['det-res'])
                utils.save_label((all_preds[0]).clone(), fp, img_name, save_conf=True, rescale=True)

            # target_nums_clean = evaluator.get_patch_pos_batch(all_preds)[0]
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
    cfg.DATA.AUGMENT = 0
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
        print(cmd)
        os.system(cmd)

        # (det-results)Test attack performance with detections as GT.
        # GT label: detections on clean samples.
        # Target detection: detections on adversarial samples.
        # print('ground truth     :', os.path.join(path, MAP_PATHS['det-lab']))
        det_mAP = compute_mAP(path=path, ignore=args.ignore_class, lab_path=MAP_PATHS['attack-lab'],
                                gt_path=MAP_PATHS['det-lab'], res_prefix='det', quiet=quiet)
        det_mAPs[detector.name] = round(det_mAP*100, 2)

        if hasattr(args, 'test_gt') or hasattr(args, 'test_origin'):
            # soft link the path of the GT labels
            gt_target = os.path.join(path, 'ground-truth')
            gt_source = os.path.join(args.label_path, MAP_PATHS['ground-truth'] + label_postfix)
            path_remove(gt_target)
            cmd = ' '.join(['ln -s ', gt_source, gt_target])
            print(cmd)
            os.system(cmd)

            if args.test_gt:
                # (gt-results) Test attack performance with annotations as GT.
                # GT label: annotations.
                # Target detection: detections on adversarial samples.
                gt_mAP = compute_mAP(path=path, ignore=args.ignore_class, lab_path=MAP_PATHS['attack-lab'],
                                        gt_path=MAP_PATHS['ground-truth'], res_prefix='gt', quiet=quiet)
                gt_mAPs[detector.name] = round(gt_mAP*100, 2)

            if args.test_origin:
                # (ori-results) Test original performance of the detector.
                # GT label: annotations.
                # Target detection: detections on clean samples.
                ori_mAP = compute_mAP(path=path, ignore=args.ignore_class, lab_path=MAP_PATHS['det-res'],
                                      gt_path=MAP_PATHS['ground-truth'], res_prefix='ori', quiet=quiet)
                ori_mAPs[detector.name] = round(ori_mAP * 100, 2)

    return det_mAPs, gt_mAPs, ori_mAPs


def save_results(results, results_file):
    if not os.path.exists(results_file):
        # write the header if the det_mAP_file is firstly created.
        with open(results_file, 'a') as f:
            f.write(logger_msg('model name', 'mAP'))
            f.write('--------------------------\n')
    dict2txt(results, results_file)

if __name__ == '__main__':
    from utils.parser import dict2txt, logger_msg
    # To test attack performance with reference to detection labels.

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, default=None, help="Optional. Init by an given patch.")
    parser.add_argument('-cfg', '--cfg', type=str, default=None, help="A RELATIVE file path of teh .yaml cfg file.")
    parser.add_argument('-s', '--save', type=str, default=os.path.join(PROJECT_DIR, 'data/inria/'), help="Directory to save evaluation results. ** Use an ABSOLUTE path.")
    parser.add_argument('-lp', '--label_path', type=str, default=os.path.join(PROJECT_DIR, 'data/INRIAPerson/Test/labels'), help='Directory ground truth & detection labels. ** Use an ABSOLUTE path')
    parser.add_argument('-dr', '--data_root', type=str, default=os.path.join(PROJECT_DIR, 'data/INRIAPerson/Test/pos'), help='Directory of the target image data to evaluate. ** Use an ABSOLUTE path.')
    parser.add_argument('-to', '--test_origin', action='store_true', help="Test performance of the detectors in clean samples.")
    parser.add_argument('-tg', '--test_gt', action='store_true', help="Test attack performance based on ground truth labels(Annotation).")
    parser.add_argument('-ul', '--stimulate_uint8_loss', action='store_true', help="Stimulate uint8 loss from float preprocesser format.")
    parser.add_argument('-i', '--save_imgs', help='to save attacked imgs', action='store_true')
    parser.add_argument('-ng', '--gen_no_label', action='store_true', help="Won't generate any detection labels of adversarial samples if set True.")
    parser.add_argument('-e', '--eva_class', type=str, default='0', help="The class to attack. '-1': all classes, '-2': attack seen classes(ATTACK_CLASS in cfg file), '-3': attack unseen classes(all_class - ATTACK_CLASS); or custom '0, 2:5, 10'.")
    parser.add_argument('-q', '--quiet', action='store_true', help='Verbose none if set true.')
    args = parser.parse_args()

    args.save = os.path.abspath(args.save)
    args.label_path = os.path.abspath(args.label_path)
    args.data_root = os.path.abspath(args.data_root)

    cfg = ConfigParser(args.cfg)
    args = get_save(args)
    # args, evaluator = eval_init(args, cfg)
    det_mAPs, gt_mAPs, ori_mAPs = eval_patch(args, cfg)

    # save mAP results to .txt results file.
    save_results(det_mAPs, os.path.join(args.save, 'det-mAP.txt'))
    if args.test_gt:
        save_results(gt_mAPs, os.path.join(args.save, 'gt-mAP.txt'))
    if args.test_origin:
        dict2txt(ori_mAPs, os.path.join(args.save, 'ori-mAP.txt'))

    if not args.quiet:
        logger_msg("Attack Performance. AP", det_mAPs)
        logger_msg("See results in path", args.save)

