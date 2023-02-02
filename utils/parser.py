import numpy as np
import yaml
import time

from .utils import obj
import sys, os
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)


def load_class_names(namesfile, trim=True):
    # namesfile = self.DATA.CLASS_NAME_FILE
    all_class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        # print('line: ', line)
        if trim:
            line = line.replace(' ', '')
        all_class_names.append(line)
    return all_class_names


class ConfigParser:
    """To parse a proj config .yaml file.
        * Storing information of class names and basic configs.
        * Process the attacking class list.

    """
    def __init__(self, config_file):
        self.config_file = config_file
        # self.all_class_names = []
        self.attack_list = []
        self.load_config()
        self.all_class_names = load_class_names(
            os.path.join(PROJECT_DIR, self.DATA.CLASS_NAME_FILE))
        self.get_attack_list()
        self.num_classes = len(self.all_class_names)

        # TODO: To be more universal. This is a temp var(Attack single cls).
        self.attack_cls = int(self.ATTACKER.ATTACK_CLASS)

    def get_attack_list(self):
        self.attack_list = self.rectify_class_list(self.ATTACKER.ATTACK_CLASS, dtype='int')
        # print('Attack class list from config file: ', self.attack_list)

    def rectify_class_list(self, class_str_list, dtype='int', split_str = ':'):
        class_list = []
        if class_str_list == '-1':
            # attack all classes
            print('Attack all classes')
            class_list = self.all_class_names
            if dtype == 'int':
                class_list = list(range(0, len(self.all_class_names)))
            return class_list
        elif class_str_list == '-2':
            # attack seen classes(ATTACK_CLASS in cfg file)
            print('Evaluating in the seen classes')
            if dtype == 'int':
                return self.attack_list
            elif dtype == 'str':
                return self.show_class_label(self.attack_list)
        elif class_str_list == '-3':
            # attack unseen classes(all_classes - ATTACK_CLASS)
            print('Evaluating in the unseen classes')
            if dtype == 'int':
                all_class_int = list(range(0, len(self.all_class_names)))
                return list(set(all_class_int).difference(set(self.attack_list)))
            elif dtype == 'str':
                return list(set(self.all_class_names).difference(set(self.show_class_label(self.attack_list))))
        else:
            # attack given attack list like a:b / 0 / 0, 1, 2
            # import ast
            # class_str_list = ast.literal_eval(class_str_list)
            # print(class_str_list)
            class_str_list = class_str_list.split(', ')
            # print(class_str)
            class_list = []
            for class_str in class_str_list:
                if isinstance(class_str, str) and split_str in class_str:
                    # class 'a:b': [a, b)
                    left, right = class_str.split(split_str)
                    class_list = list(range(int(left), int(right)))
                else:
                    class_list.append(int(class_str))

            if dtype == 'str':
                class_list = self.show_class_label(class_list)

        return class_list

    def show_class_index(self, class_names):
        class_index = np.zeros(len(class_names), dtype=int)
        cur = 0
        for index, name in enumerate(self.all_class_names):
            if name in class_names:
                class_index[cur] = index
                cur += 1
        assert cur == len(class_names), 'Error!'
        return class_index.tolist()

    def show_class_label(self, class_list):
        if -1 in class_list:
            return self.all_class_names

        names = []
        for class_id in class_list:
            name = self.all_class_names[class_id]
            names.append(name.replace(' ', ''))
        # print('Class names: ', names)
        return names

    def load_config(self):
        cfg = yaml.load(open(self.config_file), Loader=yaml.FullLoader)
        for a, b in cfg.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)

    def __str__(self):
        pass


def ignore_class(eva_args, cfg):
    # Be careful of the so-called 'attack_list' and 'eva_class' in the evaluate.py
    # For higher reusability of the codes, these variable names may be confusing
    # In this file, the 'attack_list' is loaded from the config file which has been used for training
    # (cuz we don't bother to create a new config file for evaluations)
    # Thus the 'attack_list' refers to the original attacked classes when training the patch
    # while the 'eva_list' denotes the class list to be evaluated, which are to attack in the evaluation
    # (When the eva classes are different from the original attack classes,
    # it is mainly for the partial attack in evaluating unseen-class/cross-class performance)
    eva_args.eva_class_list = cfg.rectify_class_list(eva_args.eva_class, dtype='str')
    # print('Eva(Attack) classes from evaluation: ', cfg.show_class_index(args.eva_class_list))
    # print('Eva classes names from evaluation: ', args.eva_class_list)

    eva_args.ignore_class = list(set(cfg.all_class_names).difference(set(eva_args.eva_class_list)))
    if len(eva_args.ignore_class) == 0: eva_args.ignore_class = None
    return eva_args


def logger_msg(k, v):
    msg = '{:>30} : {:<30}'.format(str(k), str(v))
    print(msg)
    return msg

def logger_banner(banner):
    dot = '------------------------------------------------------------'
    pos = int(len(dot)/2 - len(banner)/2)
    banner = dot[:pos] + banner + dot[pos+len(banner):]
    print(banner)


def logger_cfg(cfg, banner=None):
    if banner is not None:
        logger_banner(banner)
    for k, v in cfg.__dict__.items():
        if isinstance(v, obj):
            logger_cfg(v)
        else:
            logger_msg(k, v)


def logger(cfg, args):
    logger_banner('Training')
    logger_msg('cfg', args.cfg)
    localtime = time.asctime(time.localtime(time.time()))
    logger_msg('time', localtime)
    logger_cfg(cfg.DATA, 'DATA')
    logger_cfg(cfg.DETECTOR, 'DETECTOR')
    logger_cfg(cfg.ATTACKER, 'ATTACKER')
    logger_banner('END')


def merge_dict_by_key(dict_s, dict_d):
    for k, v in dict_s.items():
        dict_d[k] = [v, dict_d[k]]
    return dict_d


def dict2txt(dict, filename, ljust=16):
    with open(filename, 'a') as f:
        for k, v in dict.items():
            f.write(k.ljust(ljust, ' ') + ':\t' + str(v))
            f.write('\n')

