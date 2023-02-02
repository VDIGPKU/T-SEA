import torch

from .PyTorch_YOLOv3.pytorchyolo.models import load_model
from .PyTorch_YOLOv3.pytorchyolo.utils.utils import rescale_boxes, non_max_suppression
from ...base import DetectorBase


class HHYolov3(DetectorBase):
    def __init__(self,
                 name, cfg, input_tensor_size=412,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_size, device)
        self.target = None

    def requires_grad_(self, state: bool):
        self.detector.module_list.requires_grad_(state)
    
    def load(self, model_weights, detector_config_file=None):
        self.detector = load_model(model_path=detector_config_file, weights_path=model_weights).to(self.device)
        self.eval()

    def __call__(self, batch_tensor: torch.tensor, **kwargs):
        detections_with_grad = self.detector(batch_tensor) # torch.tensor([1, num, classes_num+4+1])
        preds = non_max_suppression(detections_with_grad, self.conf_thres, self.iou_thres)
        obj_confs = detections_with_grad[:, :, 4]
        cls_max_ids = detections_with_grad[:, :, 5]

        bbox_array = []
        for i, pred in enumerate(preds):
            box = rescale_boxes(pred, self.input_tensor_size, self.ori_size)
            box[:, [0, 2]] /= self.ori_size[1]
            box[:, [1, 3]] /= self.ori_size[0]
            box[:, :4] = torch.clamp(box[:, :4], min=0, max=1)
            # print(box)
            bbox_array.append(box)

        output = {'bbox_array': bbox_array, 'obj_confs': obj_confs, "cls_max_ids": cls_max_ids}
        return output