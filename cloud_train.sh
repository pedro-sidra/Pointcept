set -e

pip install huggingface-cli wandb
mkdir -p /workspace/Pointcept/data/
cd /workspace/Pointcept/data/
wandb artifact get --type dataset --root . pedrosidra/dataset/scannet:v0&
huggingface-cli download --repo-type dataset  --local-dir . --  Pointcept/arkitscenes-compressed
for i in 1 2 3 4 5 6 7 8; do tar -xzvf arkitscenes_$i.tar.gz; done
mv Training/ scannet/arkit/

cd /workspace/Pointcept
bash scripts/train.sh -g 2 -d scannet -c pipeline-sculpting-spunet-arkit -n icip/sculpt_multi

