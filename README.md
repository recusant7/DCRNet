## Overview

This is the  Pytorch implementation of "Dilated convolution based CSI Feedback Compression
for Massive MIMO"

## Requirements

To use this project, you need to ensure the following requirements are installed.

- Python >= 3.6
- Pytorch == 3.7 
- tqdm
- colorama
## Dataset
The used COST2100 dataset can be found in the paper of  [CsiNet+](https://ieeexplore.ieee.org/abstract/document/8972904/) and the corresponding [GoogDrive](https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj).
## Train 
```python3
python main.py --gpu 0 --lr 2e-3 -v 5 --cr 4 --scenario "in" --expansion 1
```
or ignore the output 
```python3
nohup python main.py --gpu 0 --lr 2e-3 -v 5 --cr 4 --scenario "in" --expansion 1  > /dev/null 2>&1 &
```
The training logs and checkpoints are saved in ```./outputs```

## checkpoints
The checkpoints will be available in GoogleDrive soon. 