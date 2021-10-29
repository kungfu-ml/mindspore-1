#!/bin/bash

# export CUDA_VISIBLE_DEVICES=2,3

RANK_SIZE=1
EPOCH_SIZE=1
DATA_DIR="${HOME}/data"
REPO_DIR="${HOME}/Elasticity/Repo/kungfu-mindspore"

export KUNGFU_CONFIG_LOG_LEVEL="DEBUG"
export KUNGFU_CONFIG_ENABLE_STALL_DETECTION="true"

. ${REPO_DIR}/ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path ${REPO_DIR}/mindspore)
export KUNGFU_NO_AUTO_INIT=1
export NCCL_P2P_LEVEL="NVL"

CKPT_DIR="./checkpoint"
if [ -d ${CKPT_DIR} ]; then
    rm -rf ${CKPT_DIR}
fi
mkdir ${CKPT_DIR}
cp ${DATA_DIR}/bert/bert_base_squad.ckpt ${CKPT_DIR}/model.ckpt

# python -m kungfu.cmd.elastic_run \
kungfu-run \
    -np $RANK_SIZE \
    -logfile kungfu-run.log \
    -logdir ./log \
    -port-range 10500-11000 \
    -w \
    -config-server http://127.0.0.1:9100/config \
    -builtin-config-port 9100 \
    python run_squad_baseline_elastic.py \
        --device_target="GPU" \
        --distribute="true" \
        --do_train="true" \
        --do_eval="false" \
        --device_id=0 \
        --epoch_num=${EPOCH_SIZE} \
        --num_class=2 \
        --train_data_shuffle="true" \
        --eval_data_shuffle="false" \
        --train_batch_size=32 \
        --eval_batch_size=1 \
        --vocab_file_path="${DATA_DIR}/bert/bert_uncased_L-12_H-768_A-12/vocab.txt" \
        --save_finetune_checkpoint_path="${CKPT_DIR}" \
        --load_pretrain_checkpoint_path="${CKPT_DIR}/model.ckpt" \
        --train_data_file_path="${DATA_DIR}/squad1/train.tf_record" \
        --eval_json_path="${DATA_DIR}/squad1/dev-v1.1.json" \
        --schema_file_path="${DATA_DIR}/squad1/squad_schema.json" > squad.log 2>&1
