import numpy as np
import torchvision
import torch
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DET_LIB = os.path.join(PROJECT_DIR, 'detlib')
sys.path.append(PROJECT_DIR)
from utils.parser import logger_msg

def init_detectors(cfg_det: object=None, distribute: bool =False):
    detector_names = cfg_det.NAME
    detectors = []
    if distribute:
        assert torch.cuda.device_count() >= len(detector_names), \
            'available device should bigger than num_detectors'
        for i, detector_name in enumerate(detector_names):
            detector = init_detector(detector_name, cfg_det, device=torch.device(f'cuda:{i}'))
            detectors.append(detector)
    else:
        for detector_name in detector_names:
            detector = init_detector(detector_name, cfg_det)
            detectors.append(detector)
    return detectors


def init_detector(detector_name: str, cfg: object, device: torch.device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    detector = None
    detector_name = detector_name.lower()
    model_config = cfg.MODEL_CONFIG if hasattr(cfg, 'MODEL_CONFIG') else None

    if detector_name == "yolov2":
        from detlib.HHDet import HHYolov2
        detector = HHYolov2(name=detector_name, cfg=cfg, device=device)
        if model_config is None:
            model_config = 'HHDet/yolov2/yolov2/config/yolo.cfg'
        detector.load(
            detector_config_file=os.path.join(DET_LIB, model_config),
            model_weights=os.path.join(DET_LIB, 'HHDet/yolov2/yolov2/weights/yolo.weights')
        )

    elif detector_name == "yolov3":
        from detlib.HHDet import HHYolov3
        detector = HHYolov3(name=detector_name, cfg=cfg, device=device)
        if model_config is None:
            model_config = 'HHDet/yolov3/PyTorch_YOLOv3/config/yolov3.cfg'
        if cfg.PERTURB.GATE == 'shakedrop':
            print('Self-ensemble! Shakedrop ')
            model_config = 'HHDet/yolov3/PyTorch_YOLOv3/config/yolov3-chr.cfg'
        detector.load(
            detector_config_file=os.path.join(DET_LIB, model_config),
            model_weights=os.path.join(DET_LIB, 'HHDet/yolov3/PyTorch_YOLOv3/weights/yolov3.weights'),
        )

    elif detector_name == "yolov3-tiny":
        from detlib.HHDet import HHYolov3
        detector = HHYolov3(name=detector_name, cfg=cfg, device=device)
        if model_config is None:
            model_config = 'HHDet/yolov3/PyTorch_YOLOv3/config/yolov3-tiny.cfg'
        detector.load(
            detector_config_file=os.path.join(DET_LIB, model_config),
            model_weights=os.path.join(DET_LIB, 'weights/yolov3-tiny.weights'))

    elif detector_name == "yolov4-tiny":
        from detlib.HHDet import HHYolov4
        detector = HHYolov4(name=detector_name, cfg=cfg, device=device)
        if model_config is None:
            model_config = 'HHDet/yolov4/Pytorch_YOLOv4/cfg/yolov4-tiny.cfg'
        detector.load(
            detector_config_file=os.path.join(DET_LIB, model_config),
            model_weights=os.path.join(DET_LIB, 'weights/yolov4-tiny.weights'))

    elif detector_name == "yolov4":
        from detlib.HHDet import HHYolov4
        detector = HHYolov4(name=detector_name, cfg=cfg, device=device)

        if cfg.PERTURB.GATE == 'shakedrop':
            print('Self-ensemble! Shakedrop v4')
            model_config = 'HHDet/yolov4/Pytorch_YOLOv4/cfg/yolov4-shakedrop.cfg'
        elif model_config is None:
            model_config = 'HHDet/yolov4/Pytorch_YOLOv4/cfg/yolov4.cfg'

        detector.load(
            detector_config_file=os.path.join(DET_LIB, model_config),
            model_weights=os.path.join(DET_LIB, 'HHDet/yolov4/Pytorch_YOLOv4/weight/yolov4.weights')
        )

    elif detector_name == "yolov5":
        from detlib.HHDet import HHYolov5
        detector = HHYolov5(name=detector_name, cfg=cfg, device=device)
        if cfg.PERTURB.GATE == 'shakedrop':
            model_config = 'HHDet/yolov5/yolov5/models/yolov5s-shakedrop.yaml'
        elif cfg.PERTURB.GATE == 'ghostshake':
            model_config = 'HHDet/yolov5/yolov5/models/yolov5s-ghostshake.yaml'
        elif model_config is None:
            model_config = 'HHDet/yolov5/yolov5/models/yolov5s.yaml'

        detector.load(
            model_weights=os.path.join(DET_LIB, 'HHDet/yolov5/yolov5/weight/yolov5s.pt'),
            model_config=os.path.join(DET_LIB, model_config)
        )

    elif detector_name == "ssd" or detector_name == 'ssdlite':
        from detlib.torchDet import TorchSSD
        detector = TorchSSD(name=detector_name, cfg=cfg, device=device)
        detector.load()
        # detector.load('./checkpoints/ssd300_coco_20210803_015428-d231a06e.pth')
        # detector.load(os.path.join(DET_LIB, 'HHDet/ssd/ssd_pytorch/weights/vgg16_reducedfc.pth'))

    elif detector_name == "faster_rcnn":
        from detlib.torchDet import TorchFasterRCNN
        detector = TorchFasterRCNN(detector_name, cfg, device=device)
        detector.load()

    elif detector_name == 'center_net':
        from detlib.AfreeDet import CenterNet
        arch = cfg.ARCH if hasattr(cfg, 'ARCH') else 'resdcn_18'
        model_weights = cfg.weights if hasattr(cfg, 'weights') else 'weights/ctdet_coco_resdcn18.pth'
        detector = CenterNet(detector_name, cfg, device=device)
        detector.load(os.path.join(DET_LIB, model_weights), arch=arch)

    logger_msg('model cfg', model_config)
    return detector


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names
