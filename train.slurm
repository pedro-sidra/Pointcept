#!/bin/bash
#SBATCH --job-name=pcept_train_debugging
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

MASTER_HOSTNAME=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
MASTER_ADDR=$(host $MASTER_HOSTNAME | awk '/has address/ { print $4 }')
DATASET=scannet
EXP_NAME=test
MASTER_PORT=23450
# DIST_URL=tcp://$MASTER_ADDR:$MASTER_PORT
DIST_URL='auto'

echo $DIST_URL
echo $SCRATCH

# mkdir -p $SCRATCH/exp
# cd $HOME/Pointcept && docker compose run -v $SCRATCH/exp:/workspaces/Pointcept/exp -e DIST_URL=$DIST_URL --rm dev
# cd $HOME/Pointcept && docker compose run -v $SCRATCH/exp:/workspaces/Pointcept/exp -e SLURM_NODEID=$SLURM_NODEID -e DIST_URL=$DIST_URL --rm dev bash scripts/train.sh -g 1 -d scannet -c semseg-spunet-sidra-efficient-lr10 -n debug/slurm_sclr10 -m 2
# cd $HOME/Pointcept && docker compose run -v $SCRATCH/exp:/workspaces/Pointcept/exp -e SLURM_NODEID=$SLURM_NODEID -e DIST_URL=$DIST_URL --rm dev bash scripts/train.sh -g 1 -d scannet -c pipeline-trimming-spunet -n debug/trimming
cd $HOME/Pointcept && docker compose run -v $SCRATCH/exp:/workspaces/Pointcept/exp -e SLURM_NODEID=$SLURM_NODEID -e DIST_URL=$DIST_URL --rm dev bash scripts/train.sh -g 1 -d scannet -c pipeline-sculpting-sonata -n debug/ptv3_slurm