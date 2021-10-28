# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

'''
Bert finetune and evaluation script.
'''
import argparse
import collections
import os

import mindspore.common.dtype as mstype
import mindspore.ops.operations.kungfu_comm_ops as kfops
from kungfu.python.elastic import create_tf_records
from mindspore import context
from mindspore import log as logger
from mindspore._c_expression import kungfu_nccl_finalize, kungfu_nccl_init
from mindspore.common import set_seed
from mindspore.common.tensor import Tensor
from mindspore.nn.optim import AdamWeightDecay, Lamb, Momentum
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import (CheckpointConfig, ModelCheckpoint,
                                      SummaryCollector, TimeMonitor)
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.bert_for_finetune import BertSquad, BertSquadCell
from src.callback import (CheckpointCallback, ElasticScheduleCallback,
                          KungFuSummaryCallback, DebugCallback, GlobalStepProgressCallback)
from src.dataset import create_squad_dataset
from src.elastic_state import ElasticCallback, ElasticState
from src.finetune_eval_config import bert_net_cfg, optimizer_cfg
from src.kungfu_mindspore_optimizer import KungFuLamb
from src.utils import (BertLearningRate, LoadNewestCkpt, LossCallBack,
                       make_directory)

_cur_dir = os.getcwd()

# HACK
DROPPED = 0
GLOBAL_BATCH_SIZE = 0
SEED = 1


def do_train(dataset=None, network=None, load_checkpoint_path="", save_checkpoint_path="",
             epoch_num=1, distributed=False):
    """ do train """
    if load_checkpoint_path == "":
        raise ValueError("Pretrain model missed, finetune task must load pretrain model!")
    #  steps_per_epoch = dataset.get_dataset_size()
    steps_per_epoch = 2770 # HARDCODED
    print("Dataset size {}".format(dataset.get_dataset_size())) # DEBUGGING
    print("Optimiser {}".format(optimizer_cfg.optimizer)) # DEBUGGING
    # optimizer
    if optimizer_cfg.optimizer == 'AdamWeightDecay':
        lr_schedule = BertLearningRate(learning_rate=optimizer_cfg.AdamWeightDecay.learning_rate,
                                       end_learning_rate=optimizer_cfg.AdamWeightDecay.end_learning_rate,
                                       warmup_steps=int(steps_per_epoch * epoch_num * 0.1),
                                       decay_steps=steps_per_epoch * epoch_num,
                                       power=optimizer_cfg.AdamWeightDecay.power)
        params = network.trainable_params()
        decay_params = list(filter(optimizer_cfg.AdamWeightDecay.decay_filter, params))
        other_params = list(filter(lambda x: not optimizer_cfg.AdamWeightDecay.decay_filter(x), params))
        group_params = [{'params': decay_params, 'weight_decay': optimizer_cfg.AdamWeightDecay.weight_decay},
                        {'params': other_params, 'weight_decay': 0.0}]

        optimizer = AdamWeightDecay(group_params, lr_schedule, eps=optimizer_cfg.AdamWeightDecay.eps)
    elif optimizer_cfg.optimizer == 'Lamb':
        lr_schedule = BertLearningRate(learning_rate=optimizer_cfg.Lamb.learning_rate,
                                       end_learning_rate=optimizer_cfg.Lamb.end_learning_rate,
                                       warmup_steps=int(steps_per_epoch * epoch_num * 0.1),
                                       decay_steps=steps_per_epoch * epoch_num,
                                       power=optimizer_cfg.Lamb.power)
        optimizer = KungFuLamb(network.trainable_params(), learning_rate=lr_schedule)
    elif optimizer_cfg.optimizer == 'Momentum':
        optimizer = Momentum(network.trainable_params(), learning_rate=optimizer_cfg.Momentum.learning_rate,
                             momentum=optimizer_cfg.Momentum.momentum)
    else:
        raise Exception("Optimizer not supported. support: [AdamWeightDecay, Lamb, Momentum]")

    # load checkpoint into network
    ckpt_config = CheckpointConfig(save_checkpoint_steps=500, keep_checkpoint_max=10)
    #  ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix="squad",
                                 directory=None if save_checkpoint_path == "" else save_checkpoint_path,
                                 config=ckpt_config)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(network, param_dict)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2**32, scale_factor=2, scale_window=1000)
    netwithgrads = BertSquadCell(network, optimizer=optimizer, scale_update_cell=update_cell)
    model = Model(netwithgrads)

    #  callbacks = [TimeMonitor(dataset.get_dataset_size()), LossCallBack(dataset.get_dataset_size()), ckpoint_cb]
    callbacks = []

    # Summary (loss)
    if distributed:
        rank = kfops.kungfu_current_rank()
        summary_path = "./summary_{}.csv".format(rank)
    else:
        summary_path = "./summary.csv"
    callbacks.append(KungFuSummaryCallback(summary_path))

    # ELASTIC
    max_progress = 88641
    print("max_progress {}".format(max_progress))
    es = ElasticState(max_progress - DROPPED, True)

    path = "./checkpoint"
    callbacks.append(CheckpointCallback(es, model, path))

    callbacks.append(GlobalStepProgressCallback(model, es, GLOBAL_BATCH_SIZE))

    #  schedule = {8320: 2, 22400: 1, 32640: 2, 38400: 1, 46080: 2, 64640: 1, 75520: 2}
    schedule = {5120: 2, 12800: 1, 23680: 2, 30080: 1, 40320: 2, 67840: 1, 79360: 2}
    print("schedule {}".format(schedule))
    schedule_cb = ElasticScheduleCallback(es, schedule, model)
    callbacks.append(schedule_cb)
    callbacks.append(ElasticCallback(es, GLOBAL_BATCH_SIZE))

    callbacks.append(DebugCallback(model))

    model.train(epoch_num,
                dataset,
                callbacks=callbacks,
                dataset_sink_mode=False)


def do_eval(dataset=None, load_checkpoint_path="", eval_batch_size=1):
    """ do eval """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    net = BertSquad(bert_net_cfg, False, 2)
    net.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(net, param_dict)
    model = Model(net)
    output = []
    RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])
    columns_list = ["input_ids", "input_mask", "segment_ids", "unique_ids"]
    for data in dataset.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, segment_ids, unique_ids = input_data
        start_positions = Tensor([1], mstype.float32)
        end_positions = Tensor([1], mstype.float32)
        is_impossible = Tensor([1], mstype.float32)
        logits = model.predict(input_ids, input_mask, segment_ids, start_positions,
                               end_positions, unique_ids, is_impossible)
        ids = logits[0].asnumpy()
        start = logits[1].asnumpy()
        end = logits[2].asnumpy()

        for i in range(eval_batch_size):
            unique_id = int(ids[i])
            start_logits = [float(x) for x in start[i].flat]
            end_logits = [float(x) for x in end[i].flat]
            output.append(RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits))
    return output

def run_squad():
    """run squad task"""
    parser = argparse.ArgumentParser(description="run squad")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["Ascend", "GPU"],
                        help="Device type, default is Ascend")
    parser.add_argument("--distribute", type=str, default="false", choices=["true", "false"],
                        help="Run distribute, default is false.")
    parser.add_argument("--do_train", type=str, default="false", choices=["true", "false"],
                        help="Eable train, default is false")
    parser.add_argument("--do_eval", type=str, default="false", choices=["true", "false"],
                        help="Eable eval, default is false")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--epoch_num", type=int, default=1, help="Epoch number, default is 1.")
    parser.add_argument("--num_class", type=int, default=2, help="The number of class, default is 2.")
    parser.add_argument("--train_data_shuffle", type=str, default="true", choices=["true", "false"],
                        help="Enable train data shuffle, default is true")
    parser.add_argument("--eval_data_shuffle", type=str, default="false", choices=["true", "false"],
                        help="Enable eval data shuffle, default is false")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Train batch size, default is 32")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Eval batch size, default is 1")
    parser.add_argument("--vocab_file_path", type=str, default="", help="Vocab file path")
    parser.add_argument("--eval_json_path", type=str, default="", help="Evaluation json file path, can be eval.json")
    parser.add_argument("--save_finetune_checkpoint_path", type=str, default="", help="Save checkpoint path")
    parser.add_argument("--load_pretrain_checkpoint_path", type=str, default="", help="Load checkpoint file path")
    parser.add_argument("--load_finetune_checkpoint_path", type=str, default="", help="Load checkpoint file path")
    parser.add_argument("--train_data_file_path", type=str, default="",
                        help="Data path, it is better to use absolute path")
    parser.add_argument("--schema_file_path", type=str, default="",
                        help="Schema path, it is better to use absolute path")
    args_opt = parser.parse_args()
    epoch_num = args_opt.epoch_num
    load_pretrain_checkpoint_path = args_opt.load_pretrain_checkpoint_path
    save_finetune_checkpoint_path = args_opt.save_finetune_checkpoint_path
    load_finetune_checkpoint_path = args_opt.load_finetune_checkpoint_path

    if args_opt.do_train.lower() == "false" and args_opt.do_eval.lower() == "false":
        raise ValueError("At least one of 'do_train' or 'do_eval' must be true")
    if args_opt.do_train.lower() == "true" and args_opt.train_data_file_path == "":
        raise ValueError("'train_data_file_path' must be set when do finetune task")
    if args_opt.do_eval.lower() == "true":
        if args_opt.vocab_file_path == "":
            raise ValueError("'vocab_file_path' must be set when do evaluation task")
        if args_opt.eval_json_path == "":
            raise ValueError("'tokenization_file_path' must be set when do evaluation task")

    """ distributed """
    if args_opt.distribute.lower() == "true":
        distributed = True
    else:
        distributed = False
    if distributed:
        kfops.init(args_opt.device_target)
        kungfu_nccl_init()
        device_num = kfops.kungfu_current_cluster_size()
        rank = kfops.kungfu_current_rank()
        print("kungfu rank={}, size={}".format(rank, device_num))

        save_finetune_checkpoint_path = os.path.join(save_finetune_checkpoint_path,
                                                     "ckpt_" + str(rank))
    else:
        device_num = 1
        rank = 0

    target = args_opt.device_target
    if target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)
    elif target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        #  context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
        if bert_net_cfg.compute_type != mstype.float32:
            logger.warning('GPU only support fp32 temporarily, run with fp32.')
            bert_net_cfg.compute_type = mstype.float32
    else:
        raise Exception("Target error, GPU or Ascend is supported.")

    netwithloss = BertSquad(bert_net_cfg, True, 2, dropout_prob=0.1)

    # ELASTICITY
    home = os.getenv("HOME")
    index_path = os.path.join(home, "data/squad1/tf-index-1.idx.txt")
    global GLOBAL_BATCH_SIZE
    GLOBAL_BATCH_SIZE = args_opt.train_batch_size
    print("before create_tf_records")
    shard = create_tf_records(index_path, SEED, GLOBAL_BATCH_SIZE)
    filenames = shard['filenames']
    print("file names {}".format(filenames))
    batch_size, _ = shard['batch_sizes'][0]
    global DROPPED
    DROPPED = shard['dropped']

    if args_opt.do_train.lower() == "true":
        ds = create_squad_dataset(batch_size=batch_size, repeat_count=1,
                                  data_file_path=filenames,
                                  schema_file_path=args_opt.schema_file_path,
                                  do_shuffle=(args_opt.train_data_shuffle.lower() == "true"),
                                  device_num=device_num, rank=rank)

        do_train(ds, netwithloss, load_pretrain_checkpoint_path, save_finetune_checkpoint_path,
                epoch_num, distributed)
        if args_opt.do_eval.lower() == "true":
            if save_finetune_checkpoint_path == "":
                load_finetune_checkpoint_dir = _cur_dir
            else:
                load_finetune_checkpoint_dir = make_directory(save_finetune_checkpoint_path)
            load_finetune_checkpoint_path = LoadNewestCkpt(load_finetune_checkpoint_dir,
                                                           ds.get_dataset_size(), epoch_num, "squad")

    if args_opt.do_eval.lower() == "true":
        from src import tokenization
        from src.create_squad_data import (convert_examples_to_features,
                                           read_squad_examples)
        from src.squad_get_predictions import write_predictions
        from src.squad_postprocess import SQuad_postprocess
        tokenizer = tokenization.FullTokenizer(vocab_file=args_opt.vocab_file_path, do_lower_case=True)
        eval_examples = read_squad_examples(args_opt.eval_json_path, False)
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=bert_net_cfg.seq_length,
            doc_stride=128,
            max_query_length=64,
            is_training=False,
            output_fn=None,
            vocab_file=args_opt.vocab_file_path)
        ds = create_squad_dataset(batch_size=args_opt.eval_batch_size, repeat_count=1,
                                  data_file_path=eval_features,
                                  schema_file_path=args_opt.schema_file_path, is_training=False,
                                  do_shuffle=(args_opt.eval_data_shuffle.lower() == "true"),
                                  device_num=device_num, rank=rank)
        outputs = do_eval(ds, load_finetune_checkpoint_path, args_opt.eval_batch_size)
        all_predictions = write_predictions(eval_examples, eval_features, outputs, 20, 30, True)

        if distributed:
            output_path = "./output_{}.json".format(rank)
        else:
            output_path = "./output.json"

        SQuad_postprocess(args_opt.eval_json_path, all_predictions, output_metrics=output_path)

    kfops.finalize(args_opt.device_target)
    kungfu_nccl_finalize()


if __name__ == "__main__":
    set_seed(SEED)
    run_squad()
