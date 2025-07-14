#!/bin/bash
#SBATCH --job-name=pcept_train
#SBATCH --partition=tupi
#SBATCH --exclude=tupi[1-2]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm/%x/%j_%N.out
#SBATCH --error=slurm/%x/%j_%N.err

# mkdir -p $SCRATCH/exp
DIST_URL='auto'
DATASET=scannet
CONFIG=semseg-spunet-sidra-efficient-lr100
# CONFIG=pipeline-trimming-spunet
EXP_NAME="wacv/scratch_spunet_lr100_v2"
WEIGHT="None"
MACHINES=1
cd $HOME/Pointcept && docker compose run -v $SCRATCH/exp:/workspaces/Pointcept/exp -e SLURM_NODEID=$SLURM_NODEID -e DIST_URL=$DIST_URL --rm dev bash scripts/train.sh -g 1 -d $DATASET  -m $MACHINES -c $CONFIG -n $EXP_NAME -w $WEIGHT
