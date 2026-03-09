# test lasher
#CUDA_VISIBLE_DEVICES=1 NCCL_P2P_LEVEL=NVL python ./RGBT_workspace/test_rgbt_mgpus.py --script_name dmtrack --dataset_name LasHeR --yaml_name rgbt

# test rgbt234
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_LEVEL=NVL python ./RGBT_workspace/test_rgbt_mgpus.py --script_name dmtrack --dataset_name RGBT234 --yaml_name rgbt 