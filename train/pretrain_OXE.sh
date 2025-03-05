srun -p mozi-S1 -n8  --gres=gpu:8  --ntasks-per-node=8 \
    python -u train/slurm_deepspeed_train.py \
        --model UniAct_05B_CodeBook_256_V1_Fulltune \
        --recipe oxe_magic_soup \
        --iters 1000000 \
        --start_iters 0 \
        --initial_t 2.0 \
        --final_t 0.5 \
        --batch-size 32 \
        --lr 1e-5 \
        --grad_accumulation_steps 1 \
        --output_dir exp/pretrain \
        --save_interval 10000 \
        --precision bf16 