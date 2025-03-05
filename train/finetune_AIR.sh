srun -p mozi-S1 -n8  --gres=gpu:8  --ntasks-per-node=8   \
    python -u slurm_deepspeed_train.py \
        --model UniAct_05B_CodeBook_256_V1_Fulltune \
        --recipe ACT-Libero \
        --iters 100000 \
        --start_iters 0 \
        --batch-size 32 \
        --initial_t 1.4 \
        --final_t 1.4 \
        --lr 1e-4 \
        --grad_accumulation_steps 1 \
        --output_dir exp/Libero-resume \
        --save_interval 10000 \
        --port 12345 \
        --seed 178945 \
        --precision bf16 \
        --resume "your path here"