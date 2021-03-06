mkdir 'log'
python train.py \
    --data_root  \
    --train_file '/export2/wangjun492/face_database/facex-zoo/share_file/train_data/MS-Celeb-1M-v1c-r-shallow_train_list.txt' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file '../backbone_conf.yaml' \
    --head_type 'SST_Prototype' \
    --head_conf_file '../head_conf.yaml' \
    --lr 0.1 \
    --out_dir 'out_dir' \
    --epoches 250 \
    --step '150,200,230' \
    --print_freq 100 \
    --batch_size 512 \
    --momentum 0.9 \
    --alpha 0.999 \
    --log_dir 'log' \
    --tensorboardx_logdir 'sst_mobileface' \
    --save_freq 10 \
    2>&1 | tee log/log.log
