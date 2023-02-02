## Framework Overview

Three individual core modules: Attack, Detlib & Utils. An overview: 
* **Detlib**
Detlib is the detection module, which implements the input and output interfaces for individual detectors as well as an agent for unified detector calls.
Model perturbation(e.g. Shakedrop) function is achieved and implemented inside detector module.
  * **HHDet** (PyTorch) - Yolo V2, V3, V3-tiny, V4, V4tiny, V5
    * See [**Acknowledgements**](#Acknowledgements) for introduction of the models.
    * Note that we've modified certain parts of the original code version to fit our modules (and to add functions).
  * **TorchDet** (PyTorch) - Faster RCNN(renet50), ssd(vgg16) & ssdlite(mobilenet v3 large)
    * Rewritten from Torch official detection models.

  * **Custom** - You can support your custom models based on this framework. See more details in [**README**](https://github.com/VDIGPKU/T-SEA/blob/main/detlib/README.md).


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
