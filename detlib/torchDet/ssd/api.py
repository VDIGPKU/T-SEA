import sys

import torch
from ...base import DetectorBase
from .. import inter_nms
from .ssd import ssd300_vgg16
from .ssdlite import ssdlite320_mobilenet_v3_large

model_dict = {
    'ssd': ssd300_vgg16, # with no ResBlock in the backbone
    'ssdlite': ssdlite320_mobilenet_v3_large # with ResBlock in the backbone
}


class TorchSSD(DetectorBase):
    def __init__(self, name, cfg,
                 input_tensor_size=None,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_size, device)
        self.max_conf_num = 200

    def load(self, model_weights=None, **kargs):
        if self.cfg.PERTURB.GATE == 'shakedrop':
            from .ssd import ssdlite320_mobilenet_v3_large_shakedrop
            self.detector = ssdlite320_mobilenet_v3_large_shakedrop(pretrained=True)
        else:
            self.detector = model_dict[self.name](pretrained=True)

        self.detector = self.detector.to(self.device)
        self.detector.eval()
        self.detector.requires_grad_(False)

    def __call__(self, batch_tensor, **kwargs):
        shape = batch_tensor.shape[-2]
        preds = self.detector(batch_tensor)

        # print(confs[0])
        cls_max_ids = None
        confs_array = None
        bbox_array = []
        for ind, pred in enumerate(preds):
            len = pred['scores'].shape[0]
            array = torch.cat((
                pred['boxes']/shape,
                pred['scores'].view(len, 1),
                (pred['labels']-1).view(len, 1)
            ), 1) if len else torch.cuda.FloatTensor([])

            conf = pred['scores']
            if conf.size(0) < self.max_conf_num:
                conf = torch.cat((conf, torch.zeros(self.max_conf_num - conf.size(0)).to(self.device)), -1)
            confs_array = conf if confs_array is None else torch.vstack((confs_array, conf))
            bbox_array.append(array)

        bbox_array = inter_nms(bbox_array, self.conf_thres, self.iou_thres)
        output = {'bbox_array': bbox_array, 'obj_confs': confs_array, "cls_max_ids": cls_max_ids}
        return output