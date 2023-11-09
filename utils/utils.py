import torch
import os
import shutil
import logging
try:
    from .convertor import FormatConverter
except:
    from convertor import FormatConverter


def save_tensor(target_tensor, save_name, save_path='./'):
    os.makedirs(save_path, exist_ok=True)
    save_target = os.path.join(save_path, save_name)

    if save_name.split('.')[-1] == 'pth':
        torch.save(target_tensor, save_target)
    else:
        if target_tensor.ndim == 4:
            target_tensor = target_tensor.squeeze(0)
        FormatConverter.tensor2PIL(target_tensor).save(save_target)


class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, obj(b) if isinstance(b, dict) else b)


def set_logging(name=None, verbose=True):
    # Sets level and returns logger
    for h in logging.root.handlers:
        logging.root.removeHandler(h)  # remove all handlers associated with the root logger object
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    logging.basicConfig(format="%(message)s", level=logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)


def getLogger():
    return set_logging(__name__)  # define globally (used in train.py, val.py, detect.py, etc.)


def path_remove(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path) # a dir
        except:
            os.remove(path) # a symbolic link


def dir_check(save_path, child_paths, rebuild=False):
    from scripts.dict import MAP_PATHS
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