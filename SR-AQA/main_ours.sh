# '--fs_layer 1 1 1 0 0' means replacing the $1- 3^{rd}$ batch normalization layers of ResNet-50 with the UFS.
# '--lambda_scl' means the lambda for SCL loss, if '--lambda_scl' > 0, then using SCL loss.
# '--lambda_tl' means the lambda for TL loss, if '--lambda_tl' > 0, then using TL loss.
# '--t_data_ratio X' means using X-tenths of unlabeled real data for training.

# For CP
nohup python main_ours.py /raid/jlxu/TEE \
        -d TEE -s S -t R --task_type cp_reg --epochs 100 -i 400 --gpu_id cuda:0 --lr 0.0001 \
        -b 32 --log logs/ours/TEE_cp --resize-size 224 --fs_layer 1 1 1 0 0 --lambda_scl 1 --lambda_tl 1 --t_data_ratio 10 >/dev/null 2>&1 &

# For GI
nohup python main_ours.py /raid/jlxu/TEE \
        -d TEE -s S -t R --task_type gi_reg --epochs 100 -i 400 --gpu_id cuda:0 --lr 0.0001 \
        -b 32 --log logs/ours/TEE_gi --resize-size 224 --fs_layer 1 1 1 0 0 --lambda_scl 1 --lambda_tl 1 --t_data_ratio 10 >/dev/null 2>&1 &
