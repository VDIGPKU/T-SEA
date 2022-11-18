import sys
import torch

from .yolov2.darknet import Darknet
from .yolov2.utils import get_region_boxes, inter_nms
from ...base import DetectorBase


class HHYolov2(DetectorBase):
    def __init__(self,
                 name, cfg,
                 input_tensor_size=412,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_size, device)

    def load(self, model_weights, detector_config_file=None):
        self.detector = Darknet(detector_config_file).to(self.device)
        self.detector.load_weights(model_weights)
        self.eval()

    def detect_test(self, batch_tensor):
        detections_with_grad = self.detector(batch_tensor)
        return detections_with_grad

    def __call__(self, batch_tensor, **kwargs):
        detections_with_grad = self.detector(batch_tensor)  # torch.tensor([1, num, classes_num+4+1])
        # x1, y1, x2, y2, det_conf, cls_max_conf, cls_max_id
        all_boxes, obj_confs, cls_max_ids = get_region_boxes(detections_with_grad, self.conf_thres,
                                            self.detector.num_classes, self.detector.anchors,
                                            self.detector.num_anchors)
        # print(all_boxes[0])
        all_boxes = inter_nms(all_boxes, conf_thres=self.conf_thres, iou_thres=self.iou_thres)
        # print(all_boxes[0])
        obj_confs = obj_confs.view(batch_tensor.size(0), -1)
        cls_max_ids = cls_max_ids.view(batch_tensor.size(0), -1)
        bbox_array = []
        for boxes in all_boxes:
            # boxes = torch.cuda.FloatTensor(boxes)
            # pad_size = self.max_n_labels - len(boxes)
            # boxes = F.pad(boxes, (0, 0, 0, pad_size), value=0).unsqueeze(0)
            if len(boxes):
                boxes[:, :4] = torch.clamp(boxes[:, :4], min=0., max=1.)
            # print(boxes.shape)
            bbox_array.append(boxes)
            # bbox_array = torch.vstack((bbox_array, boxes)) if bbox_array is not None else boxes
        # print(bbox_array)

        output = {'bbox_array': bbox_array, 'obj_confs': obj_confs, "cls_max_ids": cls_max_ids}
        return output