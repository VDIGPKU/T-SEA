import torch
import numpy as np
import cv2
import os
import sys
import argparse
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, fasterrcnn_resnet50_fpn

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)
from PIL import Image
from tools.convertor import FormatConverter
from detlib.utils import init_detector
from tools.det_utils import plot_boxes_cv2
from tools.parser import ConfigParser
import time


def get_time():
    t = time.time()
    return str(t)


def detect_gbr_im(model, cv2_im, cfg, device: torch.device=None):
    """

    :param model: detection model(in device).
    :param cv2_im: cv2 format of GBR image in random size
    :param cfg: yaml config file.
    :param device: torch.device
    :return:
    """
    if device is None:
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    original_size = cv2_im.shape[:2]
    x = cv2.resize(cv2_im, cfg.DETECTOR.INPUT_SIZE)
    rgb_x = FormatConverter.bgr_numpy2tensor(x).to(device)
    # Make detections
    results = model(rgb_x)['bbox_array']
    det_img = plot_boxes_cv2(x, FormatConverter.tensor2_numpy_cpu(results[0]), cfg.all_class_names)
    det_img = cv2.resize(det_img, original_size[::-1])
    return det_img


@torch.no_grad()
def demo(cfg, save_path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), save_frame=False):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path+'/frame/', exist_ok=True)

    model = init_detector(cfg.DETECTOR.NAME[0], cfg.DETECTOR, device=device)
    # Capture OBS Virtual Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Detectiuon Loop
    while cap.isOpened():
        ret, frame = cap.read()
        det_im = detect_gbr_im(model, frame, cfg, device)
        if save_frame:
            name = get_time() + '.png'
            cv2.imwrite(save_path+'/frame/'+name, frame)
            cv2.imwrite(save_path+name, det_im)
        cv2.imshow("Real time detection", det_im)
        # visualizaion_but_show(results, frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # a = np.zeros((800, 1200))
    # cv2.imwrite('./a.png', a)
    # exit()
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--cfg', type=str, default='baseline/v3.yaml')
    parser.add_argument('-s', '--save_path', type=str, default='./captrue/')
    args = parser.parse_args()
    args.cfg = './configs/' + args.cfg
    cfg = ConfigParser(args.cfg)
    args.save_path += cfg.DETECTOR.NAME[0] + '/'
    demo(cfg, args.save_path)
    # visualize_one_image()
