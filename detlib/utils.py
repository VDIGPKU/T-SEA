import numpy as np
import torchvision
import torch
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DET_LIB = os.path.join(PROJECT_DIR, 'detlib')
sys.path.append(PROJECT_DIR)
from utils.parser import logger_msg
from detlib.HHDet import HHYolov2, HHYolov3, HHYolov4, HHYolov5
from detlib.torchDet import TorchFasterRCNN, TorchSSD


def init_detectors(cfg_det: object=None):
    detector_names = cfg_det.NAME
    detectors = []
    for detector_name in detector_names:
        detector = init_detector(detector_name, cfg_det)
        detectors.append(detector)
    return detectors


def init_detector(detector_name: str, cfg: object, device: torch.device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    detector = None
    detector_name = detector_name.lower()
    model_config = cfg.MODEL_CONFIG if hasattr(cfg, 'MODEL_CONFIG') else None

    if detector_name == "yolov2":
        detector = HHYolov2(name=detector_name, cfg=cfg, device=device)
        if model_config is None:
            model_config = 'HHDet/yolov2/yolov2/config/yolo.cfg'
        detector.load(
            detector_config_file=os.path.join(DET_LIB, model_config),
            model_weights=os.path.join(DET_LIB, 'HHDet/yolov2/yolov2/weights/yolo.weights')
        )

    elif detector_name == "yolov3":
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
        detector = HHYolov3(name=detector_name, cfg=cfg, device=device)
        if model_config is None:
            model_config = 'HHDet/yolov3/PyTorch_YOLOv3/config/yolov3-tiny.cfg'
        detector.load(
            detector_config_file=os.path.join(DET_LIB, model_config),
            model_weights=os.path.join(DET_LIB, 'weights/yolov3-tiny.weights'))

    elif detector_name == "yolov4-tiny":
        detector = HHYolov4(name=detector_name, cfg=cfg, device=device)
        if model_config is None:
            model_config = 'HHDet/yolov4/Pytorch_YOLOv4/cfg/yolov4-tiny.cfg'
        detector.load(
            detector_config_file=os.path.join(DET_LIB, model_config),
            model_weights=os.path.join(DET_LIB, 'weights/yolov4-tiny.weights'))

    elif detector_name == "yolov4":
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
        detector = HHYolov5(name=detector_name, cfg=cfg, device=device)
        if cfg.PERTURB.GATE == 'shakedrop':
            model_config = 'HHDet/yolov5/yolov5/models/yolov5s-shakedrop.yaml'
        elif model_config is None:
            model_config = 'HHDet/yolov5/yolov5/models/yolov5s.yaml'

        detector.load(
            model_weights=os.path.join(DET_LIB, 'HHDet/yolov5/yolov5/weight/yolov5s.pt'),
            model_config=os.path.join(DET_LIB, model_config)
        )

    elif detector_name == "ssd" or detector_name == 'ssdlite':
        detector = TorchSSD(name=detector_name, cfg=cfg, device=device)
        detector.load()
        # detector.load('./checkpoints/ssd300_coco_20210803_015428-d231a06e.pth')
        # detector.load(os.path.join(DET_LIB, 'HHDet/ssd/ssd_pytorch/weights/vgg16_reducedfc.pth'))

    elif detector_name == "faster_rcnn":
        detector = TorchFasterRCNN(detector_name, cfg, device=device)
        detector.load()

    logger_msg('model cfg', model_config)
    return detector


def inter_nms(all_predictions, conf_thres: float =0.25, iou_thres: float =0.45):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    max_det = 300  # maximum number of detections per image
    out = []
    for predictions in all_predictions:
        # for each img in batch
        # print('pred', predictions.shape)
        if not predictions.shape[0]:
            out.append(predictions)
            continue
        if type(predictions) is np.ndarray:
            predictions = torch.from_numpy(predictions)
        # print(predictions.shape[0])
        try:
            scores = predictions[:, 4]
        except Exception as e:
            print(predictions.shape)
            assert 0 == 1
        i = scores > conf_thres

        # filter with conf threshhold
        boxes = predictions[i, :4]
        scores = scores[i]

        # filter with iou threshhold
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        # print('i', predictions[i].shape)
        out.append(predictions[i])
    return out


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names
