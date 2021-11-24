import os
import time

import mindspore as ms
import mindspore.ops.operations.kungfu_comm_ops as kfops
import numpy as np
from kungfu.python import current_rank, propose_new_size
from mindspore.train.serialization import save_checkpoint


class KungFuSummaryCallback(ms.train.callback.Callback):
    def __init__(self, summary_path):
        super().__init__()

        self._summary_path = summary_path

    def begin(self, run_context):
        with open(self._summary_path, "w") as summary_file:
            summary_file.write("{},{}\n".format("step", "loss"))

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        step = cb_params.cur_step_num
        output = cb_params.net_outputs
        loss = np.mean(output[0].asnumpy())

        with open(self._summary_path, "a") as summary_file:
            summary_file.write("{},{}\n".format(step, loss))


class CheckpointEveryStepCallback(ms.train.callback.Callback):
    def __init__(self, elastic_state, model, path):
        self._model = model
        self._path = path
        self._elastic_state = elastic_state

    def save_checkpoint(self):
        progress = self._elastic_state._progress
        rank = current_rank()
        ckpt_name = "model-{}-{}.ckpt".format(rank, progress)
        save_checkpoint(self._model.train_network,
                        os.path.join(self._path, ckpt_name))


    def step_end(self, run_context):
        self.save_checkpoint()

    def end(self, run_context):
        self.save_checkpoint()


class CheckpointCallback(ms.train.callback.Callback):
    def __init__(self, elastic_state, model, path):
        self._model = model
        self._path = path
        self._elastic_state = elastic_state
        self._every_progress = 1600

    def save_checkpoint(self):
        progress = self._elastic_state._progress
        rank = current_rank()
        ckpt_name = "model-{}-{}.ckpt".format(rank, progress)
        if progress % self._every_progress == 0 and progress > 0:
            save_checkpoint(self._model.train_network,
                            os.path.join(self._path, ckpt_name))


    def step_end(self, run_context):
        self.save_checkpoint()

    def end(self, run_context):
        self.save_checkpoint()


class CheckpointPowerCallback(ms.train.callback.Callback):
    def __init__(self, elastic_state, model, path):
        self._model = model
        self._path = path
        self._elastic_state = elastic_state
        self._batch_size = 32

    def is_power_bs(self, progress):
        current_value = self._batch_size
        while current_value <= progress:
            if progress == current_value:
                return True
            else:
                current_value = current_value * 2
        return False

    def save_checkpoint(self):
        progress = self._elastic_state._progress
        rank = current_rank()
        ckpt_name = "model-{}-{}.ckpt".format(rank, progress)
        if self.is_power_bs(progress):
            save_checkpoint(self._model.train_network,
                            os.path.join(self._path, ckpt_name))


    def step_end(self, run_context):
        self.save_checkpoint()

    def end(self, run_context):
        self.save_checkpoint()


class CheckpointPowerStepCallback(ms.train.callback.Callback):
    def __init__(self, model, path):
        self._model = model
        self._path = path

    def is_power_bs(self, step):
        current_value = 1
        while current_value <= step:
            if step == current_value:
                return True
            else:
                current_value = current_value * 2
        return False

    def save_checkpoint(self, step):
        rank = 0
        ckpt_name = "model-{}-{}.ckpt".format(rank, step)
        if self.is_power_bs(step):
            save_checkpoint(self._model.train_network,
                            os.path.join(self._path, ckpt_name))


    def step_end(self, run_context):
        cb_params = run_context.original_args()
        step = cb_params.cur_step_num
        self.save_checkpoint(step)

    def end(self, run_context):
        cb_params = run_context.original_args()
        step = cb_params.cur_step_num
        self.save_checkpoint(step)


def write_checkpoint(model, path, rank, step):
    ckpt_name = "model-{}-{}.ckpt".format(rank, step)
    save_checkpoint(model.train_network,
                    os.path.join(path, ckpt_name))



class DebugStepCallback(ms.train.callback.Callback):
    def print_step(self, run_context, when):
        cb_params = run_context.original_args()

        step = cb_params.cur_step_num
        print(f"{when}: run_context_step {step}")

        for param in cb_params.train_network.get_parameters():
            if param.name == "global_step":
                global_step = param.asnumpy()
                print(f"{when}: global_step {global_step}")
                break

        for param in cb_params.train_network.get_parameters():
            if param.name == "current_iterator_step":
                current_iterator_step = param.asnumpy()
                print(f"{when}: current_iterator_step {current_iterator_step}")
                break

    def step_begin(self, run_context):
        self.print_step(run_context, "begin")

    def step_end(self, run_context):
        self.print_step(run_context, "end")


class LogStepCallback(ms.train.callback.Callback):
    def __init__(self, model):
        self._model = model
        self._begin_steps = []
        self._end_steps = []
        self._begin_path = "./begin_steps.npy"
        self._end_path = "./end_steps.npy"

    def get_steps(self, run_ctx):
        cb_params = run_ctx.original_args()
        step = cb_params.cur_step_num

        params = list(self._model.train_network.get_parameters())
        global_step = None
        for param in params:
            if param.name == "global_step":
                global_step = param.asnumpy()

        return [step, int(global_step)]

    def step_begin(self, run_context):
        step_pair = self.get_steps(run_context)
        self._begin_steps.append(step_pair)
        steps_np = np.array(self._begin_steps)
        np.save(self._begin_path, steps_np)

    def step_end(self, run_context):
        step_pair = self.get_steps(run_context)
        self._end_steps.append(step_pair)
        steps_np = np.array(self._end_steps)
        np.save(self._end_path, steps_np)


class GlobalStepProgressCallback(ms.train.callback.Callback):
    def __init__(self, model, elastic_state, global_batch_size):
        self._model = model
        self._elastic_state = elastic_state
        self._global_batch_size = global_batch_size
        self._global_step_offset = self._elastic_state._progress // global_batch_size
        self._global_step = None
        self._current_iterator_step = None
        self._assign_op = ms.ops.Assign()

    def step_begin(self, run_context):
        cb_params = run_context.original_args()
        step = cb_params.cur_step_num

        if step == 1:  # adjust global step only after rescale
            for param in cb_params.train_network.get_parameters():
                if param.name == "global_step":
                    self._global_step = param
                    break
            self._assign_op(self._global_step,
                            self._global_step + self._global_step_offset)

            for param in cb_params.train_network.get_parameters():
                if param.name == "current_iterator_step":
                    self._current_iterator_step = param
                    break
            self._assign_op(self._current_iterator_step,
                            self._current_iterator_step + self._global_step_offset)


def ckpt(es):
    return 'progress-%010d.log' % (es._progress)

def save_progress(es, progress):
    with open(ckpt(es), 'w') as f:
        f.write('%d\n' % (progress))

class ElasticScheduleCallback(ms.train.callback.Callback):
    def __init__(self, es, schedule, model):
        self._es = es
        self._schedule = schedule
        self._rank = current_rank()

        if self._rank == 0:
            print('starting from progress %d' % (self._es._progress))

        self._proc_start = int(os.getenv('KUNGFU_PROC_START_TIMESTAMP'))
        self._local_step = 0

        self._model = model

    def step_begin(self, run_context):
        if self._rank == 0:
            print('running progress %d' % (self._es._progress))

        if self._rank == 0 and self._local_step == 0:
            d = time.time() - self._proc_start
            print('first step BEGIN after reload took %.fs' % (d))

        self._step_begin_ts = time.time()

    def step_end(self, run_context):
        progress = self._es._progress
        step_took = time.time() - self._step_begin_ts

        self._local_step += 1
        if self._rank == 0:
            if self._local_step == 1:
                d = time.time() - self._proc_start
                print('first step END after reload took %.fs' % (d))
            print('progress %d took %.fs' % (progress, step_took))

        if self._es._progress in self._schedule:
            if current_rank() == 0:
                new_size = self._schedule[progress]
                propose_new_size(new_size)

                save_checkpoint(self._model.train_network,
                                "./checkpoint/model.ckpt")

    def end(self, run_context):
        progress = self._es._progress
        if self._rank == 0:
            save_progress(self._es, progress)
            print('stopping at progress %d' % (progress))


class StopAfterCallback(ms.train.callback.Callback):
    def __init__(self, step):
        self._step = step

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        step = cb_params.cur_step_num
        if step == self._step:
            run_context.request_stop()


class SaveModelCallback(ms.train.callback.Callback):
    def __init__(self, model):
        self._model = model
        self._dir_path = "./params"
        self._rank = current_rank()

    def step_end(self, run_context):
        params = list(self._model.train_network.get_parameters())
        for i, param in enumerate(params):
            path = os.path.join(self._dir_path, "param-{}-{}.npy".format(self._rank , i))
            np.save(path, param.asnumpy())


class LossCallback(ms.train.callback.Callback):
    def __init__(self):
        super().__init__()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        output = cb_params.net_outputs
        loss = np.mean(output[0].asnumpy())
        print(f"loss: {loss}")
