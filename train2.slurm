#!/bin/bash
#SBATCH --job-name=multinode_pcept_train
#SBATCH --partition=tupi
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=slurm/%x/%j_%N.out
#SBATCH --error=slurm/%x/%j_%N.err
#SBATCH --exclude=tupi[1-2]

##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

# If you want to load things from your .bashrc profile, e.g. cuda drivers, singularity etc 
# source ~/.bashrc


# ******************* These are read internally it seems ***********************************
# ******** Master port, address and world size MUST be passed as variables for DDP to work 
export MASTER_PORT=$(expr 23450 + $(echo -n $SLURM_JOBID | tail -c 4) % 6)
export NCCL_PORT=$(expr 23450 + $(echo -n $SLURM_JOBID | tail -c 4 | awk '{print $1 + 1}') % 6)
GPUS=1
# export MASTER_PORT=23450
export WORLD_SIZE=$(($SLURM_NNODES * $GPUS))
echo "MASTER_PORT"=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# master_addr=$(host $master_addr | awk '/has address/ { print $4 }')
# master_addr=$(cat /etc/hosts | grep "192.* $master_addr$" | awk '{print $1}')
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
# ******************************************************************************************

# zoom zoom - recommended from lightning
# export NCCL_NSOCKS_PERTHREAD=4
# export NCCL_SOCKET_NTHREADS=2
# export NCCL_MIN_CHANNELS=32

echo "Run started at:- "
date

DATASET=scannet
EXP_NAME=debug/test_multinode
CONFIG=pipeline-sculpting-spunet
MACHINES=2
export DIST_URL=tcp://$MASTER_ADDR:$MASTER_PORT
# export DIST_URL=file:///workspaces/Pointcept/comms
    # --rm \
srun docker compose run \
    --rm --remove-orphans \
    -v $HOME/Pointcept:/workspaces/Pointcept -v $HOME/.netrc:/root/.netrc -v $SCRATCH/exp:/workspaces/Pointcept/exp \
    -e GLOO_SOCKET_IFNAME=enp4s0 \
    -e NCCL_SOCKET_IFNAME=enp4s0 \
    -e SLURM_NODELIST -e SLURM_NODEID  -e DIST_URL  -e WORLD_SIZE  -e MASTER_PORT  -e MASTER_ADDR \
    dev \
    bash scripts/train.sh -g $GPUS -d $DATASET  -m $MACHINES -c $CONFIG -n $EXP_NAME
    # -p 23450:23450 -p 23451:23451 -p 23452:23452 -p 23453:23453 -p 23454:23454 -p 23455:23455 -p 23456:23456 \
# -e SLURM_NODEID -e DIST_URL -e WORLD_SIZE -e MASTER_PORT -e MASTER_ADDR \
# --network host -w "/workspaces/Pointcept" --entrypoint "./docker-entrypoint.sh" \
    # --shm-size 32gb --gpus all \

echo ""
echo "################################################################"
echo "@@@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@"
date
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "################################################################"

