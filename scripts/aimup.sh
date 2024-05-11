#!/bin/bash
#SBATCH --job-name=Aimup
#SBATCH --partition=LocalQ
#SBATCH --output=/zhangchrai23/OperatorFormer/logs/sbatch/aimup/aimup.out
#SBATCH --error=/zhangchrai23/OperatorFormer/logs/sbatch/aimup/aimup_error.out
aim up --host 0.0.0.0 -p 6006
