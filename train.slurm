#!/bin/bash
#SBATCH --job-name=pcept_train_debugging
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

MASTER_HOSTNAME=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
MASTER_ADDR=$(host $MASTER_HOSTNAME | awk '/has address/ { print $4 }')
DATASET=scannet
EXP_NAME=test
MASTER_PORT=$((10000 + 0x$(echo -n "${DATASET}/${EXP_NAME}" | md5sum | cut -c 1-4 | awk '{print $1}') % 20000))
DIST_URL=tcp://$MASTER_ADDR:$MASTER_PORT

echo $DIST_URL
echo $SCRATCH

mkdir -p $SCRATCH/exp
# cd $HOME/Pointcept && docker compose run -v $SCRATCH/exp:/workspaces/Pointcept/exp -e DIST_URL=$DIST_URL --rm dev
cd $HOME/Pointcept && docker compose run \
    -v $SCRATCH/exp:/workspaces/Pointcept/exp --rm dev \
    # bash scripts/train.sh -g 1 -d scannet -c pipeline-sculpting-spunet -n debug/slurm
    bash scripts/train.sh -g 1 -d scannet -c semseg-spunet-sidra-efficient-lr10 -n debug/slurm_ft -w pedrosidra/debug/6a88oqzb-model_last:v19



