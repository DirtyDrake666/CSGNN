# CSGNN
This repository contains sinmle implements of the models introduced in the paper.

## Installation
The following packages were used for this project:
- ```python 3.9.13```
- ```Pytorch 2.3.1```
- ```Cuda 11.8```
- ```pyG 2.3.1```

You can follow the steps below to set up the environment
```
conda create --name csgnn python=3.9.13
conda activate csgnn
conda install pip
```
Install dependencies
```
//Pytorch install
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia

//pyG install
pip install torch_geometric

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

//Other dependencies
pip install -r requirements.txt
```

## Run Experiments
Scripts for each experiments done in the paper are located in subfolders ```experiment_name/scripts/```  

After running, the result will be in ```experiment_name/logs/``` subfolder