import mindspore as ms
import numpy as np


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
