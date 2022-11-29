#!/usr/bin/env bash
# In dir of detlib run:
# bash ./weights/setup.sh $(pwd)

proj_dir=$(pwd)
det_lib=$proj_dir/detlib

mkdir $det_lib/HHDet/yolov2/yolov2/weights
ln -s $det_lib/weights/yolov2.weights $det_lib/HHDet/yolov2/yolov2/weights/yolo.weights

mkdir $det_lib/HHDet/yolov3/PyTorch_YOLOv3/weights
ln -s $det_lib/weights/yolov3.weights $det_lib/HHDet/yolov3/PyTorch_YOLOv3/weights/yolov3.weights

mkdir $det_lib/HHDet/yolov4/Pytorch_YOLOv4/weight
ln -s $det_lib/weights/yolov4.weights $det_lib/HHDet/yolov4/Pytorch_YOLOv4/weight/yolov4.weights
ln -s $det_lib/weights/yolov4-tiny.weights $det_lib/HHDet/yolov4/Pytorch_YOLOv4/weight/yolov4-tiny.weights

mkdir $det_lib/HHDet/yolov5/yolov5/weight
ln -s $det_lib/weights/yolov5n.pt $det_lib/HHDet/yolov5/yolov5/weight/yolov5n.pt
ln -s $det_lib/weights/yolov5s.pt $det_lib/HHDet/yolov5/yolov5/weight/yolov5s.pt
ln -s $det_lib/weights/yolov5s6.pt $det_lib/HHDet/yolov5/yolov5/weight/yolov5s6.pt