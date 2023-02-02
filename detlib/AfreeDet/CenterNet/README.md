
## CenterNet Installation

```bash
cd CenterNet
pip install -r requirements.txt

# DCNv2 has supported PyTorch >= 1.0 now.
# Check in https://github.com/CharlesShang/DCNv2 and download the module which fits your env in src/models/networks.
cd src/lib/models/networks/DCNv2
bash ./make.sh
```