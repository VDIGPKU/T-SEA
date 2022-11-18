import torch

# load from YOLOV5
from .yolov5.utils.general import non_max_suppression, scale_coords
from .yolov5.models.experimental import attempt_load  # scoped to avoid circular import
from .yolov5.models.yolo import Model
from .yolov5.models.utils.general import check_yaml

from ...base import DetectorBase


class HHYolov5(DetectorBase):
    def __init__(self, name, cfg,
                 input_tensor_size=640,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_size, device)
        self.imgsz = (input_tensor_size, input_tensor_size)
        self.stride, self.pt = None, None

    def load_(self, model_weights, **args):
        w = str(model_weights[0] if isinstance(model_weights, list) else model_weights)
        self.detector = attempt_load(model_weights if isinstance(model_weights, list) else w,
                                     map_location=self.device, inplace=False)
        self.eval()
        self.stride = max(int(self.detector.stride.max()), 32)  # model stride
        self.names = self.detector.module.names if hasattr(self.detector, 'module') else self.detector.names  # get class names

    def load(self, model_weights, **args):
        model_config = args['model_config']
        # Create model
        self.detector = Model(model_config).to(self.device)
        self.detector.load_state_dict(torch.load(model_weights, map_location=self.device)['model'].float().state_dict())
        self.eval()

    def __call__(self, batch_tensor, **kwargs):
        # print('yolov5 api call')
        detections_with_grad = self.detector(batch_tensor, augment=False, visualize=False)[0]
        preds = non_max_suppression(detections_with_grad, self.conf_thres, self.iou_thres) # [batch, num, 6] e.g., [1, 22743, 1, 4]

        cls_max_ids = None
        bbox_array = []
        for pred in preds:
            box = scale_coords(batch_tensor.shape[-2:], pred, self.ori_size)
            box[:, [0,2]] /= self.ori_size[1]
            box[:, [1,3]] /= self.ori_size[0]
            bbox_array.append(box)

        # [batch, num, num_classes] e.g.,[1, 22743, 80]
        obj_confs = detections_with_grad[..., 4]
        output = {'bbox_array': bbox_array, 'obj_confs': obj_confs, "cls_max_ids": cls_max_ids}
        # exit()
        return output

