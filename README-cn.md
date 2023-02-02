# T-SEA: Transfer-based Self-Ensemble Attack on Object Detection

[**English**](https://github.com/VDIGPKU/T-SEA/blob/main/README.md)
| [**中文**](https://github.com/VDIGPKU/T-SEA/blob/main/README-cn.md)

[**Paper**](https://arxiv.org/abs/2211.09773)
| Hao Huang*, Ziyan Chen*, Huanran Chen*, Yongtao Wang, Kevin Zhang

(*共同一作)

本仓库为[T-SEA](https://arxiv.org/abs/2211.09773)
的官方源码，同时也提供了一个基于对抗补丁的通用（跨模型、跨实例）对抗攻击代码框架。

![](readme/pipeline.png)

如果本仓库对您的工作有帮助，请帮忙点亮star~ Thanks! :-D

## 更新
* 2023.01.27 - v1.2 Anchor-free检测器CenterNet已支持。
* 2022.11.22 - v1.1 修复已知的训练/测试脚本的运行bug。
* 2022.11.18 - v1.0 创建本仓库。


## 安装
### 环境
```bash
conda create -n tsea python=3.7
conda activate tsea
pip install -r requirements.txt
```

 **数据**

| 数据        |                                             检测标签                                             |                                              Source                                              |                                            
|-------------|:--------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------:|
| CCTVPerson  |  [GoogleDrive](https://drive.google.com/drive/folders/1R5DDNR0XPvSW-WyuCihDlPHf6C2XXb-o?usp=share_link)  |   [Human Detection](https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset)    |
| COCOperson  |  [GoogleDrive](https://drive.google.com/drive/folders/1zKO6yXllhReiDS04WKkb6JIkxvAW2s_9?usp=share_link)  |                            [HomePage](https://cocodataset.org/#home)                             |
| INRIAPerson |  [GoogleDrive](https://drive.google.com/drive/folders/1zKO6yXllhReiDS04WKkb6JIkxvAW2s_9?usp=share_link)  |               [Paper](https://hal.inria.fr/docs/00/54/85/12/PDF/hog_cvpr2005.pdf)                |

数据及模型权重数据详情请查阅[文档](https://github.com/VDIGPKU/T-SEA/blob/main/readme/data.md).


### 运行

**Patch Zoo** - 更多对抗补丁样本请查看[GoogleDrive](https://drive.google.com/drive/folders/1bGDf5fHVxajexKZUk22OMc5wag_adH-e?usp=share_link).

| Faster RCNN               | SSD                               | YoloV5                   |
|---------------------------|----------------------------------|--------------------------|
| ![](results/faster_rcnn-combine-demo.png) | ![](results/ssd-combine-demo.png) | ![](results/v5-demo.png) |

#### 测试

The evaluation metrics of the **Mean Average Precision([mAP](https://github.com/Cartucho/mAP))** is provided.

```bash
# 直接运行提供的示例脚本来对抗补丁样例测试
bash ./scripts/eval.sh 0 # gpu id
```

```bash
# 或运行完整命令来进行自定义测试，在项目根目录执行：

python evaluate.py \
-p ./results/v5-demo.png \
-cfg ./configs/eval/coco80.yaml \
-lp ./data/INRIAPerson/Test/labels \
-dr ./data/INRIAPerson/Test/pos \
-s ./data/test \
-e 0 # 攻击类别id

# 测试FasterRCNN、SSD(coco91): 
# 以./configs/eval/coco91.yaml作为-cfg参数运行上述命令


# 查看参数帮助
python evaluate.py -h
```

#### 对抗补丁训练
```bash
# 直接运行提供的脚本来进行一个demo对抗补丁的训练
bash ./scripts/train.sh 0 -np
# 参数: 0 gpu-id, -np 启动一个新的tensorboard进程
```

```bash
# 或者通过运行完整命令来自定义对抗补丁训练
python train_optim.py -np \
-cfg=demo.yaml \
-s=./results/demo \
-n=v5-combine-demo # 对抗补丁保存图片名&tensorboard日志文件名

# 查看参数帮助
python train_optim.py -h
```
您可以通过修改configs文件夹下的.yaml文件来自定义设置，在**configs/README.yaml**中可以查看对设置的细节介绍。


## 框架概览
我们提供了一个制作通用（跨模型&跨实例）对抗补丁的检测-攻击框架，同时提供了测试及训练模块。

框架概述请查看[文档](https://github.com/VDIGPKU/T-SEA/blob/main/readme/overview.md)
，同时对应代码模块中的README.md文件有关于模块中方法的详细阐述。

## Acknowledgements

* AdvPatch - [**Paper**](http://openaccess.thecvf.com/content_CVPRW_2019/papers/CV-COPS/Thys_Fooling_Automated_Surveillance_Cameras_Adversarial_Patches_to_Attack_Person_Detection_CVPRW_2019_paper.pdf) 
| [Source Code](https://gitlab.com/EAVISE/adversarial-yolo)

## Contact Us
If you have any problem about this work, please feel free to reach us out at `huanghao@stu.pku.edu.cn`.