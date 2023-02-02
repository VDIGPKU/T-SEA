import copy
from abc import ABC, abstractmethod
import torch

import sys, os
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

class DetectorBase(ABC):
    def __init__(self, name: str, cfg, input_tensor_size: int, device: torch.device):
        """A detector base class.

        :param name: detector title
        :param cfg: DETECTOR config for detector setting
        :param input_tensor_size: size of the tensor to input the detector. Square (x*x).
        :param device: torch device (cuda or cpu)
        """
        self.name = name
        self.device = device
        self.detector = None
        self.input_tensor_size = input_tensor_size
        self.cfg = cfg

        self.conf_thres = cfg.CONF_THRESH
        self.iou_thres = cfg.IOU_THRESH
        self.ori_size = cfg.INPUT_SIZE

        self.max_labels = 20

    @abstractmethod
    def load(self, model_weights: str, **args):
        """
        init the detector model, load trained model weights and target detection classes file
        :param model_weights
        :param **args:
            classes_path
            detector_config_file: the config file of detector
        """
        pass

    @abstractmethod
    def __call__(self, batch_tensor: torch.tensor, **kwargs):
        """
        Detection core function, get detection results by feedding the input image
        :param batch_tensor: image tensor [batch_size, channel, h, w]
        :return:
            box_array: list of bboxes(batch_size*N*6) [[x1, y1, x2, y2, conf, cls_id],..]
            detections_with_grad: confidence of the object
        """
        pass

    def requires_grad_(self, state: bool):
        """
        This highly boosts your computing by saving video memory greatly.
        Note: please rewrite it when func 'requires_grad_' cannot be called from self.detector

        :param state: require auto model gradient or not
        """
        assert self.detector, 'ERROR! Detector model not loaded yet!'
        assert state is not None, 'ERROR! Input param (state) is None!'
        self.detector.requires_grad_(state)

    def eval(self):
        """
        This is for model eval setting: fix the model and boost computing.
        """
        assert self.detector
        self.detector.eval()
        self.requires_grad_(False)

    def train(self):
        """
        This is mainly for grad perturb: model params need to be updated.
        :return:
        """
        assert self.detector
        self.detector.train()
        self.requires_grad_(True)

    def detach(self, tensor: torch.tensor):
        if self.device == torch.device('cpu'):
            return tensor.detach()
        return tensor.cpu().detach()

    def zero_grad(self):
        """
        To empty model grad.
        :return:
        """
        assert self.detector
        self.detector.zero_grad()

    def gradient_opt(self):
        assert self.cfg.PERTURB.GATE == 'grad_descend'
        self.train()
        self.ori_model = copy.deepcopy(self.detector)
        self.optimizer = torch.optim.SGD(self.detector.parameters(), lr=1e-5, momentum=0.9, nesterov=True)
        self.optimizer.zero_grad()

    def reset_model(self):
        assert self.cfg.PERTURB.GATE == 'grad_descend'
        self.detector = copy.deepcopy(self.ori_model)

    def perturb(self):
        assert self.cfg.PERTURB.GATE == 'grad_descend'
        self.optimizer.step()
        self.optimizer.zero_grad()

    def int8_precision_loss(self, img_tensor: torch.tensor):
        """
        (to stimulate the precision loss by dtype convertion in physical world)
        convert dtype of inout from float dtype to uint8 dtype, and convert back to the float dtype (including normalization)
        :param img_tensor: detached torch tensor
        :return img_tensor
        """
        # img_tensor = self.unnormalize_tensor(img_tensor)
        img_tensor *= 255.
        img_tensor = img_tensor.to(torch.uint8)
        # print(img_tensor, torch.max(img_tensor), torch.min(img_tensor))
        img_tensor = img_tensor.to(torch.float)
        img_tensor /= 255.
        # img_tensor = self.normalize_tensor(img_tensor)
        return img_tensor

    def nms(self, all_bboxes):
        from utils import inter_nms
        return inter_nms(all_bboxes, conf_thres=self.conf_thres, iou_thres=self.iou_thres)

    # def __call__(self, batch_tensor, **kwargs):
    #     original_size = (batch_tensor.size(3), batch_tensor.size(4))
    #     batch_tensor.resize_(self.input_tensor_size)
    #     self.detector.detect(batch_tensor, original_size=original_size)