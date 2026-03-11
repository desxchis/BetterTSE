python run_pretrain.py \
    --save_folder ./save/motor/pretrain_multi_weaver \
    --model_config_path configs/motor/model_multi_weaver.yaml \
    --train_config_path configs/motor/pretrain.yaml \
    --evaluate_config_path configs/motor/evaluate.yaml \
    --multipatch_num 3 \
    --L_patch_len 3 \
    --ctap_folder ./save/motor/energy/0 \
    --data_folder ./datasets/motor \
    --epochs 1000