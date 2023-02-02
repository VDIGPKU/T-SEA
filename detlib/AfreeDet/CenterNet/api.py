import sys
import torch
import numpy as np

from .CenterNet import detector_factory, opts
from ... import DetectorBase


class CenterNet(DetectorBase):
    def __init__(self, name, cfg, input_tensor_size=412,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_size, device)

    def eval(self):
        self.detector.model.requires_grad(False)
        self.detector.model.eval()

    def load(self, model_weights, **kargs):
        print('load: weights', model_weights, 'arch', kargs['arch'])
        opt = opts(load_model=model_weights, arch=kargs['arch'], input_res=self.input_tensor_size,
                   device=self.device).init()
        self.opt = opt
        self.detector = detector_factory[opt.task](opt)

        c = np.array([opt.input_w / 2., opt.input_h / 2.], dtype=np.float32)
        s = max(opt.input_h, opt.input_w) * 1.0
        self.meta = {'c': np.tile(c, (self.cfg.BATCH_SIZE, 1)),
                's': np.tile(s, (self.cfg.BATCH_SIZE, 1)),
                'out_height': self.opt.input_h // self.opt.down_ratio,
                'out_width': self.opt.input_w // self.opt.down_ratio}

    def zero_grad(self):
        self.detector.model.zero_grad()

    def __call__(self, batch_tensor, **kwargs):
        detections, bboxes = self.detector.run(batch_tensor, self.meta)
        # detections[..., :4] *= self.opt.down_ratio
        obj_confs = detections[..., 4]
        cls_max_ids = detections[..., 5]
        all_boxes = self.nms(bboxes)

        bbox_array = []
        for boxes in all_boxes:
            if len(boxes):
                boxes = boxes.detach().to(self.device)
                boxes[:, :4] = torch.clamp(boxes[:, :4]/self.input_tensor_size, min=0., max=1.)
            bbox_array.append(boxes)

        output = {'bbox_array': bbox_array, 'obj_confs': obj_confs, "cls_max_ids": cls_max_ids}
        return output