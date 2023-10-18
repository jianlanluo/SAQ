#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export D4RL_SUPPRESS_IMPORT_ERROR=1

export N_GPUS=$(nvidia-smi -L | wc -l)
export N_JOBS=16


EXP_NAME='vqiql_gym'
OUTPUT_DIR="./experiment_output"

parallel -j $N_JOBS --linebuffer --delay 3 \
    'CUDA_VISIBLE_DEVICES=$(({%} % 4)) 'python3.8 -m vqn.vqiql_main \
            --seed={1}  \
            --env={2} \
            --max_traj_length={3} \
            --eval_period=50 \
            --eval_n_trajs=20 \
            --n_pi_beta_epochs 2000 \
            --vqvae_n_epochs 500 \
            --n_epochs 1000 \
            --n_train_step_per_epoch 1000 \
            --codebook_size={4} \
            --vqvae_arch={5} \
            --sample_action={6} \
            --qf_weight_decay={7} \
            --iql_expectile={8} \
            --iql_temperature={9} \
            --kl_divergence_weight=0 \
            --logging.output_dir="$OUTPUT_DIR/$EXP_NAME" \
            --logging.online=True \
            --logging.prefix='VQN' \
            --logging.entity '' \
            --logging.project="$EXP_NAME" \
            --logging.random_delay=3.0 \
        ::: 24 42 43 \
        ::: "pen-human-v0" "door-human-v0" "hammer-human-v0" "relocate-human-v0"\
        :::+ 100 200 200 200 \
        ::: 32 64 128 \
        ::: '512-512' \
        ::: True \
        ::: 1e-3 \
        ::: 0.8 0.9 0.7 \
        ::: 1 0.5 2 \



        