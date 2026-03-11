python run_pretrain.py \
    --save_folder ./save/air/pretrain_multi_weaver \
    --model_config_path configs/air/model_multi_weaver.yaml \
    --train_config_path configs/air/pretrain.yaml \
    --evaluate_config_path configs/air/evaluate.yaml \
    --multipatch_num 3 \
    --L_patch_len 2 \
    --ctap_folder ./save/air/energy/0 \
    --data_folder ./datasets/air \
    --epochs 1000