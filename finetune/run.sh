#!/bin/bash
export OMP_NUM_THREADS=2
CUDA_VISIBLE_DEVICES="0,1"
nohup torchrun --nproc_per_node 2 -m run \
--output_dir /home/jovyan/shares/SR004.nfs2/amaksimova/exp/10/finetune/out_cool_data_bibert \
--model_name_or_path /home/jovyan/shares/SR004.nfs2/amaksimova/tune_bge_exp/models/bi_bert_pretrained \
--train_data /home/jovyan/shares/SR004.nfs2/amaksimova/exp/10/tune_cool_data1.jsonl \
--learning_rate 2e-5 \
--fp16 \
--num_train_epochs 2 \
--per_device_train_batch_size 128 \
--dataloader_drop_last True \
--query_max_len 512 \
--passage_max_len 512 \
--train_group_size 1 \
--logging_steps 100 \
--gradient_checkpointing True \
--deepspeed /home/jovyan/shares/SR004.nfs2/amaksimova/tune_bge-final/configs/ds_config.json \
--save_steps 500 \
> out_cool_data_bibert.out &
