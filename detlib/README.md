# Detlib

Detlib should support detection models and agent methods for 
unified calls like detections format parser.

We currently support detection models as:
* **HHDet**(PyTorch): Yolo V2, V3, V3-tiny, V4, V4tiny, V5
  * See **acknowledgements** in README.md in the main project directory.
* **TorchDet**(PyTorch): Faster RCNN(renet50), ssd(vgg16) & ssdlite(mobilenet v3 large)
  * Rewritten from Torch official detection models.

Model perturbation(e.g. Shakedrop) function is achieved and implemented inside detector module.

You can support your custom models by inheriting the base class and rewriting required methods.

---
### Detector Base
Main methods explained:
```python
from abc import ABC, abstractmethod

class DetectorBase(ABC):
    def __init__(self, name: str, cfg, input_tensor_size: int, device: torch.device):
        pass
    
    @abstractmethod
    def load(self, model_weights: str, **args):
        pass
    
    @abstractmethod
    def __call__(self, batch_tensor: torch.tensor, **kwargs):
        pass
    
    def requires_grad_(self, state: bool):
        # Set model.requires_grad_(False) can greatly speed up your inference.
        # Default action provided. You may need to rewrite this method if the action is different.
        assert self.detector, 'ERROR! Detector model not loaded yet!'
        assert state is not None, 'ERROR! Input param (state) is None!'
        self.detector.requires_grad_(state)
        
    def eval(self):
        # This is to fix the model.
        # Default action provided. You may need to rewrite this method if the action is different.
        assert self.detector
        self.detector.eval()
        self.requires_grad_(False)
```

---

### Pretrained models

To setup pretrained detector models.

```bash
# Run this command in the project dir
bash ./detlib/weights/download.sh
bash ./detlib/weights/setup.sh
```
Currenly only the provided models supported.

---
### Custom Detetion lib
You can support your custom detectors by providing a simple detector API.
Take the custom Yolo v3 as an example: 
```python
from base import DetectorBase

# Inherient the base class.
class HHYolov3(DetectorBase):
    def __init__(self, name, cfg, input_tensor_size=412, device):
        super().__init__(name, cfg, input_tensor_size, device)
        
    def load(self, model_weights, detector_config_file=None):
        # To load pretrained model and fix it.
        self.detector = load_weights()
        self.eval()
        pass

    def __call__(self, batch_tensor, **kwargs):
        # Obtain & format the detection results of the given image batch tensor.
        detections = self.detector()
        # Process labels as tensor([[cls_id, xmin, ymin, xmax, ymax], [], []])...
        pass

    def requires_grad_(self, state: bool):
        # Rewrite the mothed for a differnt requires_grad_ action.
        self.detector.module_list.requires_grad_(state)
```

