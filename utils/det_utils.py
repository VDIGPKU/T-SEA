# utils for detection module
import sys

import cv2
import math
import numpy as np
import torch
import torchvision
import torch.nn.functional as F


def pad_lab(lab, max_n: int, value: float = 0):
    pad_size = max_n - len(lab)
    if pad_size <= 0:
        return lab[:max_n]
    return F.pad(lab, (0, 0, 0, pad_size), value=value)


def inter_nms(all_predictions, conf_thres=0.25, iou_thres=0.45):
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
        if predictions.ndim == 1:
            predictions = predictions.unsqueeze(0)
        # print(predictions.shape[0])
        scores = predictions[:, 4]
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


def plot_boxes_cv2(imgfile, boxes, class_names, savename=None):
    """[summary]

    Args:
        imgfile ([cv2.image]): [the path of image to be detected]
        boxes ([type]): [detected boxes(list)]
        savename ([str], optional): [save image name]. Defaults to None.

    Returns:
        [cv2.image]: [cv2 type image with drawn boxes]
    """
    if type(imgfile) == type(""):
        img = cv2.imread(imgfile)
    else:
        img = imgfile
    img = np.copy(img)
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)
        bbox_thick = int(0.6 * (height + width) / 600)
        rgb = (255, 0, 0)
        if len(box) >= 6 and class_names:
            cls_conf = box[4]
            cls_id = int(box[5])
            # print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            rgb = (red, green, blue)
            if class_names[cls_id] == "":
                continue
            # print(cls_id)
            msg = str(class_names[cls_id]) + " " + str(round(cls_conf, 3))
            t_size = cv2.getTextSize(msg, 0, 0.7, thickness=bbox_thick // 2)[0]
            c1, c2 = (x1, y1), (x2, y2)
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)

            cv2.rectangle(img, (x1, y1), (np.int(c3[0]), np.int(c3[1])), rgb, -1)
            img = cv2.putText(img, msg, (c1[0], np.int(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                              bbox_thick // 2, lineType=cv2.LINE_AA)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, bbox_thick)
    if savename:
        # print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img


def rescale_boxes(boxes, current_dim, original_shape):
    """
    Rescales bounding boxes to the original shape
    Args:
        current_dim: int, denotes the size (current_dim, current_dim) of the tensor input the detector
        original_shape: int or tuple, denotes the original image size (int denotes the length of the size of a square)
    """
    if isinstance(original_shape, int):
        original_shape = (original_shape, original_shape)
    orig_h, orig_w = original_shape

    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))

    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x

    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def process_shape(x_min, y_min, x_max, y_max, ratio):
    # rectify the bbox shape into a shape of given ratio(width/height)
    # for the patches could be added with shape being fixed
    # keep the short side and trim the long side to fix the aspect ratio
    if ratio != -1:
        # print('original:', x_min, y_min, x_max, y_max)
        w = x_max - x_min
        h = y_max - y_min
        cur_ratio = w/h
        if cur_ratio > ratio:
            # width to be trimed
            trim_w = w - int(ratio * h)
            trim_w = int(trim_w / 2)
            x_min += trim_w
            x_max -= trim_w
        elif cur_ratio < ratio:
            # height to be trimed
            trim_h = h - int(w/ratio)
            trim_h = int(trim_h / 2)
            y_min += trim_h
            y_max -= trim_h

        # print('trimed:', x_min, y_min, x_max, y_max)

    return x_min, y_min, x_max, y_max


def compute_aspect_ratio(x1, y1, x2, y2, scale, aspect_ratio):
    bw = x2 - x1
    bh = y2 - y1
    target_y = math.sqrt(bw * bh * scale / aspect_ratio)
    target_x = aspect_ratio * target_y
    return target_x / 2, target_y / 2
