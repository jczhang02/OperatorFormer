---

<div align="center">

# Operator Learning for Partial Differential Equations with Attention Mechanism

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

Code for _Operator Learning for Partial Differential Equations with Attention Mechanism_.

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/jczhang02/OperatorFormer
cd OperatorFormer

# [OPTIONAL] create conda environment
conda create -n $NAME python=3.9
conda activate $NAME

# install requirements
pip install -r requirements.txt
```

## Dataset

Download dataset via [Google Drive](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-).
| Dataset Name | Download file name | Size
| ------------- | -------------- | ---------
| Burgers 1D | [Burgers_R10.zip](https://drive.google.com/file/d/16a8od4vidbiNR3WtaBPCSZ0T3moxjhYe/view?usp=drive_link) | 614.8 MB

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml

# E.g. Simple test on cpu
python src/train.py experiment=thinkpad_test

# E.g. Simple test on gpu
python src/train.py experiment=shanhe_burges
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
