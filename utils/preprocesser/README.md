# Preprocessing Before Evaluations


---

## Generate detection labels
If you want to test a victim model with clean predictions as ground truth, 
make sure you have prepared the labels corresponding to the models you want to test.

The label path tree should like the follows:
```bash
# dir-name rule: detector_name-labels
├── labels
│   ├── ground-truth-rescale-labels
│   ├── faster_rcnn-rescale-labels
│   ├── ssd-rescale-labels
│   ├── yolov3-rescale-labels
│   ├── yolov3-tiny-rescale-labels
│   ├── yolov4-rescale-labels
│   ├── yolov4-tiny-rescale-labels
│   └── yolov5-rescale-labels
```
The 'ground-truth' is the annotation labels, while the others are from clean detections of given detectors.
This is for different standards to compute mAP.

### Parse Annotations
We provide **coco_process.py**、**inria_process.py** to 
help process annotations of COCO and INRIAPerson datasets.

A demo for COCO-val2017 dataset:
```bash
# Run in the proj root dir:
python ./utils/preprocesser/coco_process.py \
--img_folder=./data/coco/val/ \
--name_file=./configs/namefiles/coco-91.names \
--json_path=./coco/instances_val2017.json \  
--save_path=./coco/val/val2017_labels/ground-truth \
--rescale_factor=416 # Decide this based on your input image size
```

Another demo for INRIAPerson dataset:
```bash
# Run in the proj root dir:
python ./utils/preprocesser/inria_process.py \
--annotations_path=./data/INRIAPerson/Test/annotations \
--save_path=./data/INRIAPerson/Test/labels/ground-truth \
--rescale_factor=416 # Decide this based on your input image size

# See help of the arguments
python ./utils/preprocesser/inria_process.py -h
```

The format of the generated label files will be like:
```bash
# cls_name x_min y_min x_max y_max
person 0.1345 0.4567 0.2456 0.9876
```

where the xyxy coordinates of the bbox is scale into `[0, 1]` or a rescaled version as `[0, input_size]`.
The latter one can meet formatting requirements of mAP.py. The rescaled label file format will be like:
```bash
# cls_name [confidence] x_min y_min x_max y_max
person [confidence] 137.2 125.0 243.5 589.6
```

### Detection labels
We provide **gen_det_labels.py** to help generate detection labels with help with our Detection-Attack framework.
```bash
# Run in the proj root dir:
python ./utils/preprocesser/gen_det_labels.py \
-dr=data/INRIAPerson/Test/pos \
-sr=data/INRIAPerson/Test/labels \
-cfg=eval/coco80.yaml
# Replace -cfg with 'eval/coco91.yaml' to generate labels from TorchDet models. 
```

python ./utils/preprocesser/gen_det_labels.py \
-dr=data/INRIAPerson/Test/pos \
-sr=./labels \
-cfg=eval/coco80.yaml