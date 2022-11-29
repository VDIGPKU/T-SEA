# T-SEA: Transfer-based Self-Ensemble Attack on Object Detection

[**English**](https://github.com/VDIGPKU/T-SEA/blob/main/README.md)
| [**中文**](https://github.com/VDIGPKU/T-SEA/blob/main/README-cn.md)


[**Paper**](https://arxiv.org/abs/2211.09773)
| Hao Huang*, Ziyan Chen*, Huanran Chen*, Yongtao Wang, Kevin Zhang

(*Equal contribution)

An official implementation of T-SEA, and also a framework provided to achieve universal (cross model&instance) patch-based adversarial attack.


![](./figures/pipeline.png)

If T-SEA is helpful for your work, please help star this repo~ Thanks! :-D

## Update
* 2022.11.22 - v1.1 Fix bugs for running train & eval scripts.
* 2022.11.18 - v1.0 This repo is created.


## Install
### Environment
```bash
conda create -n tsea python=3.7
conda activate tsea
pip install -r requirements.txt
```

### Models & Data
Make sure you have the pre-trained detector-weight files and data prepared.
* Model: Pretrained detection model weights.
* Data: Dataset of images(required) & annotations(optional, for evaluation), 
         and the corresponding detection labels (optional, generated by the victim models for evaluation).

#### Data
```bash
# Download data and prepare labels in directory data/.
# A demo file tree:
├── data
    ├── INRIAPerson
        ├── Train
        ├── Test
            ├── pos # data
            ├── labels # labels
                ├── faster_rcnn-rescale-labels
                ├── ground-truth-rescale-labels
                ├── ...

# You can generate detection labels by the given model 
# or parse the annotations based on provided util scripts.
# See more details in utils/preprocesser/README.md
```

**Download**
You can generate detection labels of your models, 
see more details in **utils/preprocessor/README.md**. 
You can also download our experimental data from [**GoogleDrive**](https://drive.google.com/drive/folders/1GzdvnLgKGiPDfitc8bIa-a76e_2Mz_Fl?usp=share_link)
| [**BaiduCloud**](https://pan.baidu.com/s/1WnjbEhYnipmGfC-TrhW-OQ?pwd=85d3). The labels and patches are included.

#### Models
You can download model weights by running:
```bash
# Run in the root proj dir.
# Download models.
bash ./detlib/weights/download.sh
# To create links to the corresponding detector modules.
bash ./detlib/weights/setup.sh
```
You can also download the supported models fom the aforementioned links.
Your file tree of the downloaded weights should like this:
```bash
# The weight file tree should like this.
└── detlib
    ├── base.py
    ├── HHDet
    ├── torchDet
    └── weights
        ├── setup.sh
        ├── yolov2.weights
        ├── yolov3-tiny.weights
        ├── ...
```

### Run
#### Evaluation
The evaluation metrics of the **Mean Average Precision(mAP)** is provided.

```bash
# You can run the demo script directly:
bash ./scripts/eval.sh 0 # gpu id
```

```bash
# To run the full command in the root proj dir:
# For yolo-models(coco80):
# Replace $PROJECT_DIR with the absolute path of your proj root dir
python evaluate.py \
-p ./results/v5-demo.png \
-cfg ./configs/eval/coco80.yaml \
-lp $PROJECT_DIR/data/INRIAPerson/Test/labels \
-dr $PROJECT_DIR/data/INRIAPerson/Test/pos \
-s $PROJECT_DIR/data/test \
-e 0 # attack class id
# for torch-models(coco91): replace -cfg with ./configs/eval/coco91.yaml

# For detailed supports of the arguments:
python evaluate.py -h
```
#### Training
```bash
# You can run the demo script directly:
bash ./scripts/train.sh 0 -np
# args: 0 gpu-id, -np new tensorboard process
```

```bash
# Or run the full command:
python train_optim.py -np \
-cfg=demo.yaml \
-s=./results/demo \
-n=v5-combine-demo # patch name & tensorboard name

# For detailed supports of the arguments:
python train_optim.py -h
```
The default save path of tensorboard logs is **runs/**.

Modify the config .yaml files for custom settings, see details in **configs/README.yaml**.


## Framework Overview
We provide a main pipeline to craft a universal adversarial patch to achieve cross-model & cross-instance attack on detectors, 
and support evaluations on given data & models.

Three individual core modules: Attack, Detlib & Utils. An overview: 
* **Detlib**
Detlib is the detection module, which implements the input and output interfaces for individual detectors as well as an agent for unified detector calls.
Model perturbation(e.g. Shakedrop) function is achieved and implemented inside detector module.
  * **HHDet** (PyTorch) - Yolo V2, V3, V3-tiny, V4, V4tiny, V5
    * See [**Acknowledgements**](#Acknowledgements) for introduction of the models.
    * Note that we've modified certain parts of the original code version to fit our modules (and to add functions).
  * **TorchDet** (PyTorch) - Faster RCNN(renet50), ssd(vgg16) & ssdlite(mobilenet v3 large)
    * Rewritten from Torch official detection models.

  * **Custom** - You can support your custom models based on this framework. See more details in **detlib/README.md**.


* **Attack Lib**
Attack Lib is the attack module, which implements the base attack methods and a core agent class for attack on detectors.
  * **base attack methods**
      * FGSM-based attack methods: **BIM**, **MIM** & **PGD**.
      * Optimizer: **SGD** & **Adam**.

* **Utils**
  * **core**
    * transformer - differentiable data transform augmentation
    * parser - config parser
    * convertor - for data formats conversion
  * **preprocessing** - for label parse and generation
  * **solver** - loss fn & schedulers
  * **metrics** - mAP 
  * **plot** - TensorBoard

See more details in the README.md file in the corresponding modules.


## Acknowledgements

### Data
* **INRIAPerson** [**Paper**](https://hal.inria.fr/docs/00/54/85/12/PDF/hog_cvpr2005.pdf)
* **COCO-person** [**HomePage**](https://cocodataset.org/#home)
* **CCTV-person** [**Source**](https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset)

### Detlib
* **HHDet**
  * Yolo V2
      [**PyTorch implementation**](https://github.com/ayooshkathuria/pytorch-yolo2)
    | [**Paper**](https://arxiv.org/abs/1506.02640)
    | [**Page**](https://pjreddie.com/darknet/yolo/)
  * Yolo V3
      [**PyTorch implementation**](https://github.com/eriklindernoren/PyTorch-YOLOv3)
    | [**Paper**](https://arxiv.org/abs/1804.02767v1)
    | [**Page**](https://pjreddie.com/darknet/yolo/)
  * Yolo V4
      [**PyTorch implementation**](https://github.com/Tianxiaomo/pytorch-YOLOv4)
    | [**Paper**](https://arxiv.org/abs/2004.10934)
    | [**Source Code**](https://github.com/AlexeyAB/darknet)
  * Yolo V5 [**PyTorch implementation**](https://github.com/ultralytics/yolov5)
* **TorchDet**
  * from **PyTorch** Detection Lib [**Docs**](https://pytorch.org/vision/0.10/models.html) | Pytorch [**Paper**](https://arxiv.org/abs/1912.01703)
  * **FasterRCNN**(resnet50 & mobilenet-v3 large) 
      [**Paper**](https://arxiv.org/abs/1506.01497)
  * **SSD** (vgg16)
      [**Paper**](https://arxiv.org/abs/1512.02325)
  * **SSDlite** (mobilenet-v3 large)
      [**Paper**](https://arxiv.org/abs/1905.02244)
  
### Attack Lib
* **Reference**: Fooling automated surveillance cameras: adversarial patches to attack person detection.
[**Source Code**](https://gitlab.com/EAVISE/adversarial-yolo)
| [**Paper**](http://openaccess.thecvf.com/content_CVPRW_2019/papers/CV-COPS/Thys_Fooling_Automated_Surveillance_Cameras_Adversarial_Patches_to_Attack_Person_Detection_CVPRW_2019_paper.pdf)

### Utils
* **Metrics**
  * mAP [**Implementation**](https://github.com/Cartucho/mAP).
* **Plot**
  * Tensorboard.

## Contact Us
If you have any problem about this work, please feel free to reach us out at `huanghao@stu.pku.edu.cn`.