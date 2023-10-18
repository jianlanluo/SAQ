#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export D4RL_SUPPRESS_IMPORT_ERROR=1


export N_GPUS=$(nvidia-smi -L | wc -l)
export N_JOBS=16


EXP_NAME='vqcql_gym'
OUTPUT_DIR="./experiment_output"

parallel -j $N_JOBS --linebuffer --delay 3 \
    'CUDA_VISIBLE_DEVICES=$(({%} % $N_GPUS)) 'python3.8 -m vqn.vqn_main \
            --seed={1}  \
            --env='{2}-{3}-v0' \
            --max_traj_length 1000 \
            --eval_period=50 \
            --vqvae_n_epochs 500 \
            --dqn_n_epochs 1000 \
            --bc_epochs 0 \
            --n_train_step_per_epoch 1000 \
            --vqn.cql_min_q_weight={4} \
            --vqn.qf_weight_decay={5} \
            --logging.output_dir="$OUTPUT_DIR/$EXP_NAME" \
            --logging.online=True \
            --logging.prefix='VQN' \
            --logging.entity '' \
            --logging.project="$EXP_NAME" \
            --logging.random_delay=3.0 \
        ::: 24 42 43 \
        ::: 'kitchen' \
        ::: 'partial' 'complete' 'mixed' \
        ::: 10 15 20 30 \
        ::: 1e-3 \



        