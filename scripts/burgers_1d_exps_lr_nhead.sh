#!/bin/bash

# Evalution the impact of different settings of lr and nhead.

python src/train.py \
	experiment=burgers \
	model.net.input_encoder_config.nhead=1,4 \
	model.optimizer.lr=1.0e-4,3.5e-4,8.0e-4 \
	task_name="Burgers_1D_exps_lr_nhead" \
	-m
