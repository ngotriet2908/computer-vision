## Installation

```
conda create -n ml-mac python=3.8.13

# Then add conda activate kaggle to the end of ~/.bashrc

source ~/.bashrc

conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

conda install pytorch-lightning=1.6.0 -c conda-forge

pip install -e .
```
