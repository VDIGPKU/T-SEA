# Attack Lib

Attack lib provide methods to craft a universal patch to achieve cross-model & cross-instance attack. 


---

### Attacker
The core attack agent to have both base attack method and detector calls.

Main methods explained:
```python
class UniversalAttacker(object):
    def __init__(self):
        pass
    
    def attack(self, img_tensor_batch, mode='sequential'):
        # Call the base attack method to optimize the patch.
        pass

    def init_attaker(self):
        pass

    def plot_boxes(self, img_tensor, boxes, save_path=None, save_name=None):
        # Plot detected boxes on images.
        pass
```
---

### Sub-modules 
  * **uap** (universal adversarial patch)
    * **Object**: Encapsulated patch object.
    * **Applier**: To apply the adversarial patch onto the image sample.
      * **Median Pool**
      * **Transformer**: To transform patch (patch augmentation including rotation, shift, jitter & cutout).
  * **methods** (base attack methods)
    * FGSM-based attack methods: **BIM**, **MIM** & **PGD**.
    * Optimizer: **SGD** & **Adam**.