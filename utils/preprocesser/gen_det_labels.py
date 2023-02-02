import os
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
import torch
import numpy as np

import sys, os
PWD = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(PWD))
sys.path.append(PROJECT_DIR)

from detlib import init_detectors
from utils import ConfigParser
from utils.loader import dataLoader
from utils.parser import load_class_names
from utils.det_utils import plot_boxes_cv2
from utils.parser import logger_msg

class Utils:
    def __init__(self, cfg):
        self.cfg = cfg
        self.class_names = load_class_names(os.path.join(PROJECT_DIR, cfg.DATA.CLASS_NAME_FILE))

    def save_label(self, preds, save_path, save_name, save_conf=True, rescale=True):
        ori_size = self.cfg.DETECTOR.INPUT_SIZE[0]
        save_name = save_name.split('.')[0] + '.txt'
        save_to = os.path.join(save_path, save_name)
        s = []
        for pred in preds:
            # N*6: x1, y1, x2, y2, conf, cls
            if rescale:
                pred[:4] *= ori_size
            x1, y1, x2, y2, conf, cls = pred
            cls = self.class_names[int(cls)].replace(' ', '')
            tmp = [cls, float(x1), float(y1), float(x2), float(y2)]
            if save_conf:
                tmp.insert(1, float(conf))
            tmp = [str(i) for i in tmp]
            s.append(' '.join(tmp))
        with open(save_to, 'w') as f:
            f.write('\n'.join(s))


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('-dr', '--data_root', type=str, default=f"data/INRIAPerson/Tes/pos", help="Image data dir path")
    parser.add_argument('-sr', '--save_root', type=str, default=f'data/INRIAPerson/Test/labels', help="Label data dir path")
    parser.add_argument('-cfg', '--config_file', type=str, default=f'advpatch/v5.yaml', help=".yaml config file, a relative path.")
    parser.add_argument('-k', '--keep_scale', action="store_true", default=False, help="To keep value range of labels as [0, 1] if set keep_scale=True. Default: rescale to the input size.")
    parser.add_argument('-i', '--imgs', action='store_true', help="To save imgs.")
    # parser.add_argument('-c', '--class', nargs='+', default=-1)
    args = parser.parse_args()

    args.data_root = os.path.join(PROJECT_DIR, args.data_root)
    args.save_root = os.path.join(PROJECT_DIR, args.save_root)
    args.config_file = os.path.join(f'{PROJECT_DIR}/configs', args.config_file)
    cfg = ConfigParser(args.config_file)
    detectors = init_detectors(cfg.DETECTOR)

    utils = Utils(cfg)
    device = torch.device('cuda')
    batch_size = 1
    img_names = [os.path.join(args.data_root, i) for i in os.listdir(args.data_root)]
    data_loader = dataLoader(data_root=args.data_root, input_size=cfg.DETECTOR.INPUT_SIZE,
                             batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    postfix = '-labels' if args.keep_scale else '-rescale-labels'

    save_path = args.save_root
    logger_msg('Dataroot', args.data_root)
    logger_msg('Rescale label', not args.keep_scale)
    logger_msg('Save dir', save_path)
    for detector in detectors:
        fp = os.path.join(save_path, detector.name + postfix)
        os.makedirs(fp, exist_ok=True)
        for index, img_tensor_batch in enumerate(tqdm(data_loader)):
            names = img_names[index:index + batch_size]
            img_name = names[0].split('/')[-1]
            all_preds = None

            img_tensor_batch = img_tensor_batch.to(detector.device)
            preds = detector(img_tensor_batch)['bbox_array']

            if args.imgs:
                save_dir = f'./test/{detector.name}'
                os.makedirs(save_dir, exist_ok=True)
                img_numpy, img_numpy_int8 = detector.unnormalize(img_tensor_batch[0])
                plot_boxes_cv2(img_numpy_int8, np.array(preds[0]), cfg.all_class_names,
                               savename=os.path.join(save_dir, img_name))
            # os.path.join('./test/' + img_name)
            # print(fp)
            utils.save_label(preds[0], fp, img_name, save_conf=False, rescale=not args.keep_scale)