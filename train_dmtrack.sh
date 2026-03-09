# Training DMTrack

# For example, RGBT
NCCL_P2P_LEVEL=NVL python tracking/train.py --script dmtrack --config rgbt  --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 0 

