#!/bin/bash

# TZ_OFFSET is the timezone offset in hours e.g. TZ_OFFSET=8 for UTC+8:00
export TZ_OFFSET=8

export NCCL_DEBUG=WARN
export WANDB_API_KEY="8000b46f9e6b884da8f0e8cc46b3c53a31711b13"
export WANDB_ENTITY=frog-ai
export WANDB_PROJECT=frogengine-test

MODEL_PATH=/mnt/public/Qwen1.5-1.8B
GLOBAL_BATCH_SIZE=16
LOCAL_BATCH_SIZE=1
WORLD_SIZE=1
GRAD_ACCU_STEPS=$(($GLOBAL_BATCH_SIZE / $LOCAL_BATCH_SIZE / $WORLD_SIZE))
RUN_NAME=sft_qwen1.5_1.8b

cd /mnt/proj/workspace/FrogEngine

ARGS="  --task sft \
        --do_train \
        --do_eval \
        --num_train_epochs 3.0 \
        --model_name_or_path $MODEL_PATH \
        --trust_remote_code True \
        --load_in_8bit True \
        --model_max_length 4096 \
        --dataset cnn_dailymail \
        --template alpaca \
        --training_mode full \
        --overwrite_cache \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps $GRAD_ACCU_STEPS \
        --gradient_checkpointing True \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --learning_rate 2e-5 \
        --lr_scheduler_type constant \
        --attention_dropout 0.1 \
        --max_grad_norm 1.0 \
        --weight_decay 1e-2 \
        --warmup_steps 100 \
        --logging_steps 1 \
        --logging_dir tensorboards/${RUN_NAME} \
        --plot_loss \
        --report_to none \
        --run_name ${RUN_NAME} \
        --output_dir outputs/${RUN_NAME} \
        --fp16"
        # --bf16 True \
        # --tf32 True \
        # bf16 and tf32 are supported by Ampere and newer GPU architectures

CUDA_VISIBLE_DEVICES=0 python src/run.py ${ARGS} 2>&1 | tee logs/${RUN_NAME}.log


# cd /mnt/proj/workspace/LLaMA-Factory
# CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
#     --stage sft \
#     --do_train \
#     --model_name_or_path path_to_llama_model \
#     --dataset alpaca_gpt4_en \
#     --template default \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj \
#     --output_dir path_to_sft_checkpoint \
#     --overwrite_cache \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --lr_scheduler_type cosine \
#     --logging_steps 10 \
#     --save_steps 1000 \
#     --learning_rate 5e-5 \
#     --num_train_epochs 3.0 \
#     --plot_loss \
#     --fp16