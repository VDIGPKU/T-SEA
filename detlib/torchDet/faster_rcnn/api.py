import torch

from . import faster_rcnn, fasterrcnn_resnet50_fpn
from ...base import DetectorBase
from .. import inter_nms


class TorchFasterRCNN(DetectorBase):
    def __init__(self, name, cfg,
                 input_tensor_size=None,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_size, device)
        self.max_conf_num = 1000

    def load(self, model_weights=None, **args):
        kwargs = {}
        if self.input_tensor_size is not None:
            kwargs['min_size'] = self.input_tensor_size
        if self.cfg.PERTURB.GATE == 'shakedrop':
            from .faster_rcnn import faster_rcnn_resnet50_shakedrop
            self.detector = faster_rcnn_resnet50_shakedrop()
        else:
            self.detector = fasterrcnn_resnet50_fpn(pretrained=True, **kwargs) \
                if model_weights is None else fasterrcnn_resnet50_fpn()

        self.detector = self.detector.to(self.device)
        self.eval()

    def __call__(self, batch_tensor, score='bbox', **kwargs):
        shape = batch_tensor.shape[-2]
        preds, confs = self.detector(batch_tensor) # the confs is scores from RPN
        bbox_array = []
        score_array = []
        for ind, (pred, now_conf) in enumerate(zip(preds, confs)):
            nums = pred['scores'].shape[0]
            array = torch.cat((
                pred['boxes'] / shape,
                pred['scores'].view(nums, 1),
                (pred['labels'] - 1).view(nums, 1)
            ), 1) if nums else torch.cuda.FloatTensor([])
            bbox_array.append(array)

            if score == "rpn":
                if now_conf.size(0) < self.max_conf_num:
                    now_conf = torch.cat((now_conf, torch.zeros(self.max_conf_num - now_conf.size(0)).to(self.device)), -1)
                    confs[ind] = now_conf
                now_conf[now_conf < 0.5] = 0
                confs[ind] = torch.mean(now_conf[now_conf > 0])
            elif score == "bbox":
                if pred['scores'].size(0) < self.max_conf_num:
                    score_array.append(torch.cat((pred['scores'], torch.zeros(self.max_conf_num - pred['scores'].size(0)).to(self.device)), -1))

        if score == "rpn":
            # score from the rpn
            confs_array = torch.vstack((confs))
        elif score == "bbox":
            # score from the final bboxes
            confs_array = torch.vstack((score_array))

        cls_max_ids = None
        bbox_array = inter_nms(bbox_array, self.conf_thres, self.iou_thres)
        output = {'bbox_array': bbox_array, 'obj_confs': confs_array, "cls_max_ids": cls_max_ids}
        return output