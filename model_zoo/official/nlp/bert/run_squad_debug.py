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
import json
import os

import mindspore.common.dtype as mstype
import mindspore.ops.operations.kungfu_comm_ops as kfops
import numpy as np
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

from src.bert_for_finetune import BertSquad, BertSquadCell, BertSquadDebug
from src.callback import (CheckpointCallback, ElasticScheduleCallback,
                          GlobalStepProgressCallback, KungFuSummaryCallback)
from src.dataset import create_squad_dataset
from src.elastic_state import ElasticCallback, ElasticState
from src.finetune_eval_config import bert_net_cfg, optimizer_cfg
from src.kungfu_mindspore_optimizer import KungFuLamb
from src.utils import BertLearningRate, LoadNewestCkpt, make_directory

_cur_dir = os.getcwd()

# HACK
DROPPED = 0
GLOBAL_BATCH_SIZE = 0
SEED = 1


def save_env_vars():
    env_dict = {}

    for k, v in os.environ.items():
        env_dict[k] = v

    with open("environment_variables.json", "w") as json_file:
        json.dump(env_dict, json_file, indent=4)


def save_python_args(args):
    arg_dict = {}

    arg_var = vars(args)
    for k, v in arg_var.items():
        arg_dict[k] = v

    with open("python_arguments.json", "w") as json_file:
        json.dump(arg_dict, json_file, indent=4)


def do_train(dataset=None, network=None, load_checkpoint_path="", save_checkpoint_path="",
             epoch_num=1):
    """ do train """
    if load_checkpoint_path == "":
        raise ValueError("Pretrain model missed, finetune task must load pretrain model!")
    #  steps_per_epoch = dataset.get_dataset_size()
    steps_per_epoch = 2770 # HARDCODED 88641//32
    print("Dataset size {}".format(dataset.get_dataset_size()))
    print("Optimiser {}".format(optimizer_cfg.optimizer))

    # optimizer
    print("=== LEARNING RATE ===")
    print("learning rate: {}".format(optimizer_cfg.Lamb.learning_rate))
    print("end learning rate: {}".format(optimizer_cfg.Lamb.end_learning_rate))
    print("step per epoch: {}".format(steps_per_epoch))
    print("number of epochs: {}".format(epoch_num))
    warmup_steps = int(steps_per_epoch * epoch_num * 0.1)
    print("warmup steps: {}".format(warmup_steps))
    decay_steps = steps_per_epoch * epoch_num
    print("decay steps: {}".format(decay_steps))
    print("power: {}".format(optimizer_cfg.Lamb.power))
    print("=== LEARNING RATE ===")
    #  lr_schedule = BertLearningRate(learning_rate=optimizer_cfg.Lamb.learning_rate,
                                   #  end_learning_rate=optimizer_cfg.Lamb.end_learning_rate,
                                   #  warmup_steps=warmup_steps,
                                   #  decay_steps=decay_steps,
                                   #  power=optimizer_cfg.Lamb.power)
    lr_schedule = optimizer_cfg.Lamb.learning_rate
    optimizer = KungFuLamb(network.trainable_params(), learning_rate=lr_schedule)
    #  from src.kungfu_mindspore_optimizer import KungFuLambDebug
    #  optimizer = KungFuLambDebug(network.trainable_params(), learning_rate=lr_schedule)

    # load checkpoint into network
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(network, param_dict)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2**32,
                                             scale_factor=2,
                                             scale_window=1000)
    netwithgrads = BertSquadCell(network,
                                 optimizer=optimizer,
                                 scale_update_cell=update_cell)
    model = Model(netwithgrads)

    # train
    from mindspore.train.dataset_helper import DatasetHelper

    rank = kfops.kungfu_current_rank()
    dataset_helper = DatasetHelper(dataset, False, -1, epoch_num)

    #  network.set_train(True)
    #  for i, batch in enumerate(dataset_helper):
        #  logits = network(*batch)
        #  np.save(f"logits-{rank}-{i}.npy", logits.asnumpy())
        #  break

    netwithgrads.set_train(True)
    for i, batch in enumerate(dataset_helper):
        logits = netwithgrads(*batch)
        #  np.save(f"logits-{rank}-{i}.npy", logits.asnumpy())
        #  if i == 9:
        break


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

    save_python_args(args_opt)

    if args_opt.do_train.lower() == "false" and args_opt.do_eval.lower() == "false":
        raise ValueError("At least one of 'do_train' or 'do_eval' must be true")
    if args_opt.do_train.lower() == "true" and args_opt.train_data_file_path == "":
        raise ValueError("'train_data_file_path' must be set when do finetune task")
    if args_opt.do_eval.lower() == "true":
        if args_opt.vocab_file_path == "":
            raise ValueError("'vocab_file_path' must be set when do evaluation task")
        if args_opt.eval_json_path == "":
            raise ValueError("'tokenization_file_path' must be set when do evaluation task")

    kfops.init(args_opt.device_target)
    kungfu_nccl_init()
    rank = kfops.kungfu_current_rank()

    save_finetune_checkpoint_path = os.path.join(save_finetune_checkpoint_path,
                                                 "ckpt_" + str(rank))

    target = args_opt.device_target
    if target == "GPU":
        #  context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
        if bert_net_cfg.compute_type != mstype.float32:
            logger.warning('GPU only support fp32 temporarily, run with fp32.')
            bert_net_cfg.compute_type = mstype.float32
    else:
        raise Exception("Target error, GPU or Ascend is supported.")

    netwithloss = BertSquad(bert_net_cfg, True, 2)
    #  netwithloss = BertSquadDebug(bert_net_cfg, True, 2)

    # ELASTICITY
    index_path = "/data/squad1/tf-index-1.idx.txt"
    global GLOBAL_BATCH_SIZE
    GLOBAL_BATCH_SIZE = args_opt.train_batch_size
    print("before create_tf_records")
    shard = create_tf_records(index_path, SEED, GLOBAL_BATCH_SIZE)
    filenames = shard['filenames']
    print("file names {}".format(filenames))
    batch_size, _ = shard['batch_sizes'][0]
    global DROPPED
    DROPPED = shard['dropped']

    ds = create_squad_dataset(batch_size=batch_size, repeat_count=1,
                              data_file_path=filenames,
                              schema_file_path=args_opt.schema_file_path,
                              do_shuffle=False)

    do_train(ds,
             netwithloss,
             load_pretrain_checkpoint_path,
             save_finetune_checkpoint_path,
             epoch_num)

    kfops.finalize(args_opt.device_target)
    kungfu_nccl_finalize()


if __name__ == "__main__":
    save_env_vars()

    set_seed(SEED)
    run_squad()
