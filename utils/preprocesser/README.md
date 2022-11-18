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

```bash
python ./utils/preprocesser/coco_process.py \
--img_folder=./preprocesser/INRIAPerson/Test/pos \
--name_file=./configs/namefiles/coco-91.names \
--json_path=./coco/instances_val2017.json \  
--save_path=./coco/val/val2017_labels/ground-truth
```
Class name files in ./config/namefiles includes coco80.names(for models with 80 prediction classes) and coco-91.names(for PyTorch model with 91 prediction classes).


The format of the generated label files will be like:
```
cls_name x_min y_min x_max y_max

person 0.1345 0.4567 0.2456 0.9876
```

where the xyxy coordinates of the bbox is scale into [0, 1].

And you are expected to further rescale labels based on **rescale_labels.py** 
to meet formatting requirements of mAP.py. The rescaled label file format will be like:
```
cls_name [confidence] x_min y_min x_max y_max

person [confidence] 137 125 243.5 589.6
```


### Detection labels
We have the **gen_det_labels.py** to help generate detection labels with help with our Detection-Attack framework.
```bash
python ./preprocesser/gen_det_labels.py \
-dr=preprocesser/coco/train/train2017 \
-sr=preprocesser/coco/train/train2017_labels \
-cfg=coco80.yaml

python ./preprocesser/gen_det_labels.py \
-dr=preprocesser/coco/train/train2017 \
-sr=preprocesser/coco/train/train2017_labels \
-cfg=coco91.yaml
```