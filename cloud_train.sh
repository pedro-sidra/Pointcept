set -e

cd /workspace/Pointcept
bash setup_datasets.sh
bash scripts/train.sh -g 2 -d scannet -c pipeline-sculpting-spunet-arkit -n icip/sculpt_multi

