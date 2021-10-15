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

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.communication.management as D
import numpy as np
from mindspore import context
from mindspore import log as logger
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore.nn.optim import AdamWeightDecay, Lamb, Momentum
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import (CheckpointConfig, LossMonitor,
                                      ModelCheckpoint, SummaryCollector,
                                      TimeMonitor)
from mindspore.train.model import Model
from mindspore.train.serialization import (load_checkpoint,
                                           load_param_into_net,
                                           save_checkpoint)

from src.bert_for_finetune import BertSquad, BertSquadCell
from src.dataset import create_squad_dataset
from src.finetune_eval_config import bert_net_cfg, optimizer_cfg
from src.utils import (BertLearningRate, LoadNewestCkpt, LossCallBack,
                       make_directory)

_cur_dir = os.getcwd()


def _set_bert_all_reduce_split():
    context.set_auto_parallel_context(parameter_broadcast=True)


def do_eval(dataset=None, load_checkpoint_path="", eval_batch_size=1):
    """ do eval """
    if load_checkpoint_path == "":
        raise ValueError(
            "Finetune model missed, evaluation task must load finetune model!")
    net = BertSquad(bert_net_cfg, False, 2)
    net.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(net, param_dict)
    model = Model(net)
    output = []
    RawResult = collections.namedtuple(
        "RawResult", ["unique_id", "start_logits", "end_logits"])
    columns_list = ["input_ids", "input_mask", "segment_ids", "unique_ids"]
    for data in dataset.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, segment_ids, unique_ids = input_data
        start_positions = Tensor([1], mstype.float32)
        end_positions = Tensor([1], mstype.float32)
        is_impossible = Tensor([1], mstype.float32)
        logits = model.predict(input_ids, input_mask, segment_ids,
                               start_positions, end_positions, unique_ids,
                               is_impossible)
        ids = logits[0].asnumpy()
        start = logits[1].asnumpy()
        end = logits[2].asnumpy()

        for i in range(eval_batch_size):
            unique_id = int(ids[i])
            start_logits = [float(x) for x in start[i].flat]
            end_logits = [float(x) for x in end[i].flat]
            output.append(
                RawResult(unique_id=unique_id,
                          start_logits=start_logits,
                          end_logits=end_logits))

    return output


def extract_rank_progress(path):
    import re

    pattern = r".*model-(\d+)*-(\d+)*.ckpt"
    match = re.search(pattern, path)
    if match is None:
        print("Regex match of entry name is None")
    rank = int(match.group(2))
    progress = int(match.group(3))

    return rank, progress


def in_tread(args_opt, dataset, checkpoints, eval_examples,
             eval_features, gpu_id):
    ms.context.set_context(device_id=gpu_id)
    from src.squad_get_predictions import write_predictions
    from src.squad_postprocess import SQuad_postprocess

    for entry in checkpoints:
        outputs = do_eval(dataset, entry.path, args_opt.eval_batch_size)
        all_predictions = write_predictions(eval_examples, eval_features,
                                            outputs, 20, 30, True)

        rank, progress = extract_rank_progress(entry.path)
        output_path = "./output_{}_{}.json".format(rank, progress)

        SQuad_postprocess(args_opt.eval_json_path,
                          all_predictions,
                          output_metrics=output_path)


def split_list(li: list, num_pieces: int):
    length = len(li)
    piece_length = length // num_pieces
    for i in range(num_pieces):
        yield li[i * piece_length: (i + 1) * piece_length]


def run_squad():
    """run squad task"""
    parser = argparse.ArgumentParser(description="run squad")
    parser.add_argument("--device_target",
                        type=str,
                        default="Ascend",
                        choices=["Ascend", "GPU"],
                        help="Device type, default is Ascend")
    parser.add_argument("--distribute",
                        type=str,
                        default="false",
                        choices=["true", "false"],
                        help="Run distribute, default is false.")
    parser.add_argument("--do_train",
                        type=str,
                        default="false",
                        choices=["true", "false"],
                        help="Eable train, default is false")
    parser.add_argument("--do_eval",
                        type=str,
                        default="false",
                        choices=["true", "false"],
                        help="Eable eval, default is false")
    parser.add_argument("--device_id",
                        type=int,
                        default=0,
                        help="Device id, default is 0.")
    parser.add_argument("--epoch_num",
                        type=int,
                        default=3,
                        help="Epoch number, default is 1.")
    parser.add_argument("--num_class",
                        type=int,
                        default=2,
                        help="The number of class, default is 2.")
    parser.add_argument("--train_data_shuffle",
                        type=str,
                        default="true",
                        choices=["true", "false"],
                        help="Enable train data shuffle, default is true")
    parser.add_argument("--eval_data_shuffle",
                        type=str,
                        default="false",
                        choices=["true", "false"],
                        help="Enable eval data shuffle, default is false")
    parser.add_argument("--train_batch_size",
                        type=int,
                        default=32,
                        help="Train batch size, default is 32")
    parser.add_argument("--eval_batch_size",
                        type=int,
                        default=1,
                        help="Eval batch size, default is 1")
    parser.add_argument("--vocab_file_path",
                        type=str,
                        default="",
                        help="Vocab file path")
    parser.add_argument("--eval_json_path",
                        type=str,
                        default="",
                        help="Evaluation json file path, can be eval.json")
    parser.add_argument("--save_finetune_checkpoint_path",
                        type=str,
                        default="",
                        help="Save checkpoint path")
    parser.add_argument("--load_pretrain_checkpoint_path",
                        type=str,
                        default="",
                        help="Load checkpoint file path")
    parser.add_argument("--load_finetune_checkpoint_path",
                        type=str,
                        default="",
                        help="Load checkpoint file path")
    parser.add_argument("--train_data_file_path",
                        type=str,
                        default="",
                        help="Data path, it is better to use absolute path")
    parser.add_argument("--schema_file_path",
                        type=str,
                        default="",
                        help="Schema path, it is better to use absolute path")
    parser.add_argument("--checkpoint_path",
                        type=str,
                        default="",
                        help="Checkpoint path")
    parser.add_argument("--num_gpus",
                        type=int,
                        default=1,
                        help="Number of GPUs")
    args_opt = parser.parse_args()
    save_finetune_checkpoint_path = args_opt.save_finetune_checkpoint_path

    if args_opt.do_train.lower() == "false" and args_opt.do_eval.lower(
    ) == "false":
        raise ValueError(
            "At least one of 'do_train' or 'do_eval' must be true")
    if args_opt.do_train.lower(
    ) == "true" and args_opt.train_data_file_path == "":
        raise ValueError(
            "'train_data_file_path' must be set when do finetune task")
    if args_opt.do_eval.lower() == "true":
        if args_opt.vocab_file_path == "":
            raise ValueError(
                "'vocab_file_path' must be set when do evaluation task")
        if args_opt.eval_json_path == "":
            raise ValueError(
                "'tokenization_file_path' must be set when do evaluation task")
    """ distributed """
    if args_opt.distribute.lower() == "true":
        distributed = True
    else:
        distributed = False
    if distributed:
        D.init()
        device_num = D.get_group_size()
        rank = D.get_rank()
        save_finetune_checkpoint_path = os.path.join(
            save_finetune_checkpoint_path, "ckpt_" + str(rank))

        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=device_num)
        _set_bert_all_reduce_split()
    else:
        device_num = 1
        rank = 0

    target = args_opt.device_target
    if target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE,
                            device_target="Ascend",
                            device_id=args_opt.device_id)
    elif target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        if bert_net_cfg.compute_type != mstype.float32:
            logger.warning('GPU only support fp32 temporarily, run with fp32.')
            bert_net_cfg.compute_type = mstype.float32
    else:
        raise Exception("Target error, GPU or Ascend is supported.")

    # EVAL
    from src import tokenization
    from src.create_squad_data import (convert_examples_to_features,
                                       read_squad_examples)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=args_opt.vocab_file_path, do_lower_case=True)
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
    dataset = create_squad_dataset(
        batch_size=args_opt.eval_batch_size,
        repeat_count=1,
        data_file_path=eval_features,
        schema_file_path=args_opt.schema_file_path,
        is_training=False,
        do_shuffle=(args_opt.eval_data_shuffle.lower() == "true"),
        device_num=device_num,
        rank=rank)

    # THREADS
    import threading
    num_gpus = args_opt.num_gpus
    entries = os.scandir(args_opt.checkpoint_path)
    entries_splits = split_list(list(entries), num_gpus)
    threads = []
    for i, split in enumerate(entries_splits):
        thr = threading.Thread(target=in_tread,
                               args=(args_opt, dataset, split, eval_examples,
                                     eval_features, i))
        thr.start()
        threads.append(thr)

    for thr in threads:
        thr.join()




if __name__ == "__main__":
    ms.common.set_seed(1)
    run_squad()
