#!/bin/bash
#SBATCH --job-name=train_lrs
#SBATCH --partition=tupi
#SBATCH --exclude=tupi[1-2]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x/%j_%N.out
#SBATCH --error=slurm/%x/%j_%N.err

DIST_URL='auto'
DATASET=scannet
CONFIG_BASE=semseg-spunet-sidra-efficient-lr
EXP_NAME_BASE="wacv/FT_trimming_spunet_lr"
WEIGHT="pedrosidra/wacv/e4zvq07p-model_last:v19"
MACHINES=1

DOCKER_ARGS="-v $SCRATCH/exp:/workspaces/Pointcept/exp -e SLURM_NODEID=$SLURM_NODEID -e DIST_URL=$DIST_URL --rm"
for LR in 1 5 10 20 100
do
    CONFIG="${CONFIG_BASE}${LR}"
    EXP_NAME="${EXP_NAME_BASE}${LR}"
    TRAIN_COMMAND="scripts/train.sh -g 1 -d $DATASET  -m $MACHINES -c $CONFIG -n $EXP_NAME -w $WEIGHT"
    srun docker compose $DOCKER_ARGS run dev bash $TRAIN_COMMAND
done
