from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import torch

class Struct:
    def __init__(self, entries):
        for k, v in entries.items():
            self.__setattr__(k, v)


def parse(opt):
    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
    opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
    opt.test_scales = [float(i) for i in opt.test_scales.split(',')]

    opt.fix_res = not opt.keep_res
    print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
    opt.reg_offset = not opt.not_reg_offset
    opt.reg_bbox = not opt.not_reg_bbox
    opt.hm_hp = not opt.not_hm_hp
    opt.reg_hp_offset = (not opt.not_reg_hp_offset) and opt.hm_hp

    if opt.head_conv == -1:  # init default head_conv
        print('arch', opt.arch)
        opt.head_conv = 256 if 'dla' in opt.arch else 64
    opt.pad = 127 if 'hourglass' in opt.arch else 31
    opt.num_stacks = 2 if opt.arch == 'hourglass' else 1

    if opt.trainval:
        opt.val_intervals = 100000000

    if opt.debug > 0:
        opt.num_workers = 0
        opt.batch_size = 1
        opt.gpus = [opt.gpus[0]]
        opt.master_batch_size = -1

    if opt.master_batch_size == -1:
        opt.master_batch_size = opt.batch_size // len(opt.gpus)
    rest_batch_size = (opt.batch_size - opt.master_batch_size)
    opt.chunk_sizes = [opt.master_batch_size]
    for i in range(len(opt.gpus) - 1):
        slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
        if i < rest_batch_size % (len(opt.gpus) - 1):
            slave_chunk_size += 1
        opt.chunk_sizes.append(slave_chunk_size)
    print('training chunk_sizes:', opt.chunk_sizes)

    opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    opt.data_dir = os.path.join(opt.root_dir, 'data')
    opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
    opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')
    print('The output will be saved to ', opt.save_dir)

    if opt.resume and opt.load_model == '':
        model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
            else opt.save_dir
        opt.load_model = os.path.join(model_path, 'model_last.pth')
    return opt


def update_dataset_info_and_set_heads(opt, dataset):
    input_h, input_w = dataset.default_resolution
    opt.mean, opt.std = dataset.mean, dataset.std
    opt.num_classes = dataset.num_classes

    # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
    input_h = opt.input_res if opt.input_res > 0 else input_h
    input_w = opt.input_res if opt.input_res > 0 else input_w
    opt.input_h = opt.input_h if opt.input_h > 0 else input_h
    opt.input_w = opt.input_w if opt.input_w > 0 else input_w
    opt.output_h = opt.input_h // opt.down_ratio
    opt.output_w = opt.input_w // opt.down_ratio
    opt.input_res = max(opt.input_h, opt.input_w)
    opt.output_res = max(opt.output_h, opt.output_w)

    if opt.task == 'exdet':
        # assert opt.dataset in ['coco']
        num_hm = 1 if opt.agnostic_ex else opt.num_classes
        opt.heads = {'hm_t': num_hm, 'hm_l': num_hm,
                     'hm_b': num_hm, 'hm_r': num_hm,
                     'hm_c': opt.num_classes}
        if opt.reg_offset:
            opt.heads.update({'reg_t': 2, 'reg_l': 2, 'reg_b': 2, 'reg_r': 2})
    elif opt.task == 'ddd':
        # assert opt.dataset in ['gta', 'kitti', 'viper']
        opt.heads = {'hm': opt.num_classes, 'dep': 1, 'rot': 8, 'dim': 3}
        if opt.reg_bbox:
            opt.heads.update(
                {'wh': 2})
        if opt.reg_offset:
            opt.heads.update({'reg': 2})
    elif opt.task == 'ctdet':
        # assert opt.dataset in ['pascal', 'coco']
        opt.heads = {'hm': opt.num_classes,
                     'wh': 2 if not opt.cat_spec_wh else 2 * opt.num_classes}
        if opt.reg_offset:
            opt.heads.update({'reg': 2})
    elif opt.task == 'multi_pose':
        # assert opt.dataset in ['coco_hp']
        opt.flip_idx = dataset.flip_idx
        opt.heads = {'hm': opt.num_classes, 'wh': 2, 'hps': 34}
        if opt.reg_offset:
            opt.heads.update({'reg': 2})
        if opt.hm_hp:
            opt.heads.update({'hm_hp': 17})
        if opt.reg_hp_offset:
            opt.heads.update({'hp_offset': 2})
    else:
        assert 0, 'task not defined!'
    print('heads', opt.heads)
    return opt

class opts(object):
    def __init__(self, task:str='ctdet', arch:str='resdcn_18', load_model='',
               input_res=-1, device=None):
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print('opts init', arch)
        self.task=task # help='ctdet | ddd | multi_pose | exdet'
        self.dataset='coco' # help='coco | kitti | coco_hp | pascal')
        self.exp_id='default'
        self.test=False
        self.debug=0    # help='level of visualization.'
              # '1: only show the final detection results'
              # '2: show the network output features'
              # '3: use matplot to display' # useful when lunching training with ipython notebook
              # '4: save all visualizations to disk'
        self.demo='' # help='path to image/ image folders/ video. "" or "webcam"')
        self.load_model=load_model # help='path to pretrained model'

        self.resume=False # help='resume an experiment. '
                  # 'Reloaded the optimizer parameter and '
                  # 'set load_model to model_last.pth '
                  # 'in the exp dir if load_model is empty.'

        # system
        self.device = device
        self.gpus='0'                   # help='-1 for CPU, use comma for multiple gpus'
        self.num_workers=4              # help='dataloader threads. 0 for single-thread.')
        self.not_cuda_benchmark=False   # help='disable when the input size is not fixed.')
        self.seed=317                   # help='random seed') # from CornerNet

        # log
        self.print_iter=0           # help='disable progress bar and print to screen.')
        self.hide_data_time =False  # help='not display time during training.')
        self.save_all=False         # help='save model to disk every 5 epochs.')
        self.metric='loss'          # help='main metric to save best model')
        self.vis_thresh=0.3         # help='visualization threshold.')
        self.debugger_theme='white' # choices=['white', 'black'])

        # model
        self.arch=arch     # help='model architecture. Currently tested'
                  # 'res_18 | res_101 | resdcn_18 | resdcn_101 |'
                  # 'dlav0_34 | dla_34 | hourglass')
        self.head_conv=-1 # help='conv layer channels for output head'
                  # '0 for no conv layer'
                  # '-1 for default setting: '
                  # '64 for resnets and 256 for dla.')
        self.down_ratio=4 # help='output stride. Currently only supports 4.')

        # input
        self.input_res=input_res # help='input height and width. -1 for default from '
                         # 'dataset. Will be overriden by input_h | input_w'
        self.input_h=-1 # help='input height. -1 for default from dataset.')
        self.input_w=-1 # help='input width. -1 for default from dataset.')

        # train
        self.lr=1.25e-4# help='learning rate for batch size 32.')
        self.lr_step='90,120' # help='drop learning rate by 10.')
        self.num_epochs=140 # help='total training epochs.')
        self.batch_size=32 # help='batch size')
        self.master_batch_size=-1 # help='batch size on the master gpu.')
        self.num_iters=-1 # help='default: #samples / batch_size.')
        self.val_intervals=5 # help='number of epochs to run validation.')
        self.trainval=False # help='include validation in training and test on test set')

        # test
        self.flip_test=False # help='flip data augmentation.')
        self.test_scales='1' # help='multi scale test augmentation.'
        self.nms=False # help='run nms in testing.'
        self.K=100 # help='max number of output objects.'
        self.not_prefetch_test=False # help='not use parallel data pre-processing.')
        self.fix_res=False # help='fix testing resolution or keep the original resolution')
        self.keep_res=False # help='keep the original resolution during validation.')

        # dataset
        self.not_rand_crop=False # help='not use the random crop data augmentation from CornerNet.')
        self.shift=0.1 # help='when not using random crop apply shift augmentation.')
        self.scale=0.4 # help='when not using random crop apply scale augmentation.')
        self.rotate=0 # help='when not using random crop apply rotation augmentation.')
        self.flip=0.5 # help='probability of applying flip augmentation.')
        self.no_color_aug=False # help='not use the color augmenation from CornerNet')
        # multi_pose
        self.aug_rot=0 # help='probability of applying rotation augmentation.')

        # ddd
        self.aug_ddd=0.5 # help='probability of applying crop augmentation.')
        self.rect_mask=False # help='for ignored object, apply mask on the rectangular region or just center point.')
        self.kitti_split='3dop' # help='different validation split for kitti: 3dop | subcnn')

        # loss
        self.mse_loss=False # help='use mse loss or focal loss to train keypoint heatmaps.')
        # ctdet
        self.reg_loss='l1' # help='regression loss: sl1 | l1 | l2')
        self.hm_weight=1 # help='loss weight for keypoint heatmaps.')
        self.off_weight=1 # help='loss weight for keypoint local offsets.')
        self.wh_weight=0.1 # help='loss weight for bounding box size.')
        # multi_pose
        self.hp_weight=1 # help='loss weight for human pose offset.')
        self.hm_hp_weight=1 # help='loss weight for human keypoint heatmap.')
        # ddd
        self.dep_weight=1                             # help='loss weight for depth.')
        self.dim_weight=1                             # help='loss weight for 3d bounding box size.')
        self.rot_weight=1                             # help='loss weight for orientation.')
        self.peak_thresh=0.2

        # task
        # ctdet
        self.norm_wh=False # help='L1(\hat(y) / y, 1) or L1(\hat(y), y)')
        self.dense_wh=False # help='apply weighted regression near center or just apply regression on center point.')
        self.cat_spec_wh=False # help='category specific bounding box size.')
        self.not_reg_offset=False # help='not regress local offset.')
        # exdet
        self.agnostic_ex=False  # help='use category agnostic extreme points.')
        self.scores_thresh=0.1     # help='threshold for extreme point heatmap.')
        self.center_thresh=0.1     # help='threshold for centermap.')
        self.aggr_weight=0.0       # help='edge aggregation weight.')
        # multi_pose
        self.dense_hp=False
                         # help='apply weighted pose regression near center or just apply regression on center point.')
        self.not_hm_hp=False
                         # help='not estimate human joint heatmap, directly use the joint offset from center.')
        self.not_reg_hp_offset=False
                         # help='not regress local offset for human joint heatmaps.')
        self.not_reg_bbox=False
                         # help='not regression bounding box size.'

        # ground truth validation
        self.eval_oracle_hm=False # help='use ground center heatmap.')
        self.eval_oracle_wh=False # help='use ground truth bounding box size.')
        self.eval_oracle_offset=False  # help='use ground truth local heatmap offset.')
        self.eval_oracle_kps=False # help='use ground truth human pose offset.')
        self.eval_oracle_hmhp=False # help='use ground truth human joint heatmaps.')
        self.eval_oracle_hp_offset=False # help='use ground truth human joint local offset.')
        self.eval_oracle_dep=False # help='use ground truth depth.')

    def init(self):
        default_dataset_info = {
          'ctdet': {'default_resolution': [512, 512], 'num_classes': 80,
                    'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                    'dataset': 'coco'},
          'exdet': {'default_resolution': [512, 512], 'num_classes': 80,
                    'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                    'dataset': 'coco'},
          'multi_pose': {
              'default_resolution': [512, 512], 'num_classes': 1,
              'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
              'dataset': 'coco_hp', 'num_joints': 17,
              'flip_idx': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                           [11, 12], [13, 14], [15, 16]]},
          'ddd': {'default_resolution': [384, 1280], 'num_classes': 3,
                  'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                  'dataset': 'kitti'},
        }

        opt = parse(self)
        dataset = Struct(default_dataset_info[opt.task])
        opt.dataset = dataset.dataset
        opt = update_dataset_info_and_set_heads(self, dataset)
        return opt
