bash scripts/train.sh -g 2 -c semseg-ptv3-sidra-efficient-lr1 -o -n icip/ptv3_lr1_sculpt -w /root/Pointcept/artifacts/ptv3_sculpt:v0/ptv3_sculpting.pth
bash scripts/train.sh -g 2 -c semseg-ptv3-sidra-efficient-lr1 -o -n icip/ptv3_lr1 
bash scripts/train.sh -g 2 -c semseg-ptv3-sidra-efficient-lr5 -o -n icip/ptv3_lr5_sculpt -w /root/Pointcept/artifacts/ptv3_sculpt:v0/ptv3_sculpting.pth
bash scripts/train.sh -g 2 -c semseg-ptv3-sidra-efficient-lr5 -o -n icip/ptv3_lr5 
bash scripts/train.sh -g 2 -c semseg-ptv3-sidra-efficient-lr10 -o -n icip/ptv3_lr10_sculpt -w /root/Pointcept/artifacts/ptv3_sculpt:v0/ptv3_sculpting.pth
bash scripts/train.sh -g 2 -c semseg-ptv3-sidra-efficient-lr10 -o -n icip/ptv3_lr10
bash scripts/train.sh -g 2 -c semseg-ptv3-sidra-efficient-lr20 -o -n icip/ptv3_lr20_sculpt -w /root/Pointcept/artifacts/ptv3_sculpt:v0/ptv3_sculpting.pth
bash scripts/train.sh -g 2 -c semseg-ptv3-sidra-efficient-lr20 -o -n icip/ptv3_lr20