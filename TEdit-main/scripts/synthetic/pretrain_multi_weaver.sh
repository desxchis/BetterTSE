python run_pretrain.py \
    --save_folder ./save/synthetic/pretrain_multi_weaver \
    --model_config_path configs/synthetic/model_multi_weaver.yaml \
    --train_config_path configs/synthetic/pretrain.yaml \
    --evaluate_config_path configs/synthetic/evaluate.yaml \
    --multipatch_num 3 \
    --L_patch_len 3 \
    --ctap_folder ./save/synthetic/energy/0 \
    --data_folder ./datasets/synthetic \
    --epochs 200