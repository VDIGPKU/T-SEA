import torch
import numpy as np
from tqdm import tqdm
import os
import time

from utils.loader import dataLoader
from utils import save_tensor
from utils.parser import logger
from scripts.dict import scheduler_factory
from train_optim import init

def modelDDP(detector_attacker, args):
    for ind, detector in enumerate(detector_attacker.detectors):
        detector_attacker.detectors[ind] = torch.nn.parallel.DistributedDataParallel(detector,
                                                                                     device_ids=[args.local_rank],
                                                                                     output_device=args.local_rank,
                                                                                     find_unused_parameters=True)


def attack(cfg, detector_attacker, save_name, args=None, data_root=None):
    get_iter = lambda epoch, index: (epoch - 1) * len(data_loader) + index
    if data_root is None: data_root = cfg.DATA.TRAIN.IMG_DIR
    data_loader, vlogger = init(detector_attacker, cfg, data_root=data_root, args=args)

    save_tensor(detector_attacker.universal_patch, save_name + '.png', args.save_path)
    scheduler = scheduler_factory[cfg.ATTACKER.LR_SCHEDULER](detector_attacker.attacker)
    loss_array = []
    for epoch in range(1, cfg.ATTACKER.MAX_EPOCH + 1):
        et0 = time.time()
        ep_loss = 0
        # for index, img_tensor_batch in enumerate(tqdm(data_loader, desc=f'Epoch {epoch}')):
        for index, img_tensor_batch in enumerate(tqdm(data_loader, desc=f'Epoch {epoch}')):
            now_step = get_iter(epoch, index)
            if vlogger: vlogger(epoch, now_step)
            img_tensor_batch = img_tensor_batch.to(detector_attacker.device)
            all_preds = detector_attacker.detect_bbox(img_tensor_batch)
            # get position of adversarial patches
            target_nums = detector_attacker.get_patch_pos_batch(all_preds)
            if sum(target_nums) == 0: continue

            loss = detector_attacker.attack(img_tensor_batch, args.attack_method)
            ep_loss += loss

            if epoch % 10 == 0:
                # patch_name = f'{epoch}_{save_name}'
                patch_name = f'{save_name}' + '.png'
                save_tensor(detector_attacker.universal_patch, patch_name, args.save_path)

        et1 = time.time()
        ep_loss /= len(data_loader)
        loss_array.append(ep_loss)
        scheduler.step(ep_loss=ep_loss, epoch=epoch)
        if vlogger:
            vlogger.write_ep_loss(ep_loss)
            vlogger.write_scalar(et1 - et0, 'misc/ep time')

    np.save(os.path.join(args.save_path, save_name + '-loss.npy'), loss_array)


if __name__ == '__main__':
    from utils.parser import ConfigParser
    from attack.attacker import UniversalAttacker
    import argparse
    import warnings
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, help='fine-tune from a pre-trained patch', default=None)
    parser.add_argument('-a', '--augment_data', action='store_true')
    parser.add_argument('-m', '--attack_method', type=str, default='sequential')
    parser.add_argument('-cfg', '--cfg', type=str, default='test.yaml')
    parser.add_argument('-n', '--board_name', type=str, default=None)
    parser.add_argument('-s', '--save_path', type=str, default='./results/inria')
    parser.add_argument('-d', '--debugging', action='store_true')
    parser.add_argument('-np', '--new_process', action='store_true', default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_patch_name = args.cfg.split('/')[-1].split('.')[0] if args.board_name is None else args.board_name
    args.cfg = './configs/' + args.cfg

    print('-------------------------Training-------------------------')
    print('                       device : ', device)
    print('                          cfg :', args.cfg)

    cfg = ConfigParser(args.cfg)
    detector_attacker = UniversalAttacker(cfg, device)
    cfg.show_class_label(cfg.attack_list)
    attack(cfg, detector_attacker, save_patch_name, args)
