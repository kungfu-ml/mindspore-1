#!/bin/bash
set -e

RANK_SIZE=4
EPOCH_SIZE=1
DATA_DIR="/data/squad1/train.tf_record"
SCHEMA_DIR="/home/marcel/Mindspore/squad_schema.json"

# export CUDA_VISIBLE_DEVICES=2,3

. /home/marcel/Mindspore/kungfu-mindspore/ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path /home/marcel/Mindspore/kungfu-mindspore/mindspore)

for i in 1 2 3
do
    /home/marcel/KungFu/kungfu/bin/kungfu-run \
        -np $RANK_SIZE \
        -logfile kungfu-run.log \
        -logdir ./log \
        -port-range 10500-11000 \
        python run_squad_kungfu.py  \
            --device_target="GPU" \
            --distribute="true" \
            --do_train="true" \
            --do_eval="true" \
            --device_id=0 \
            --epoch_num=${EPOCH_SIZE} \
            --num_class=2 \
            --train_data_shuffle="true" \
            --eval_data_shuffle="false" \
            --train_batch_size=6 \
            --eval_batch_size=1 \
            --vocab_file_path="/home/marcel/Mindspore/bert_uncased_L-12_H-768_A-12/vocab.txt" \
            --save_finetune_checkpoint_path="./checkpoint" \
            --load_pretrain_checkpoint_path="/home/marcel/Mindspore/bert_base_squad.ckpt" \
            --train_data_file_path=${DATA_DIR} \
            --eval_json_path="/data/squad1/dev-v1.1.json" \
            --schema_file_path=${SCHEMA_DIR} > squad.log 2>&1

    mv ./checkpoint /data/marcel/bert_experiments/squad_distr_24_${RANK_SIZE}_kf_seed/${i}
    mv ./log /data/marcel/bert_experiments/squad_distr_24_${RANK_SIZE}_kf_seed/${i}
    mv ./squad.log /data/marcel/bert_experiments/squad_distr_24_${RANK_SIZE}_kf_seed/${i}
    mv ./output_* /data/marcel/bert_experiments/squad_distr_24_${RANK_SIZE}_kf_seed/${i}
    mv ./summary_* /data/marcel/bert_experiments/squad_distr_24_${RANK_SIZE}_kf_seed/${i}
done
