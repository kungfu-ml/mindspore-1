import kungfu.python as kfpy
import mindspore as ms
import mindspore.dataset as ds
import mindspore.ops.operations.kungfu_comm_ops as kfops


class ElasticDataset(ds.TFRecordDataset):
    def __init__(self):
        pass

    def repartition(self):
        pass


def get_cluster_config():
    return {"cluster_config": {}}


class TrainingState():
    def __init__(self, dataset, model, optimizer):
        self._dataset = dataset
        self._model = model
        self._optimizer = optimizer
        self._state = dict()

        self._state["model_parameters"] = list(self._model.train_network.get_parameters())
        self._state["optimizer_parameters"] = list(self._optimizer.get_parameters())
        self._state["batch_size"] = self._dataset.get_batch_size()

    def get_state(self):
        return self._state


class TrainingStateManager():
    def __init__(self, training_state):
        self._training_state = training_state
        self._last_cluster_config = None

        self._broadcast_op = kfops.KungFuBroadcast()

    def check_for_change(self):
        changed, detached = kfpy.resize()
        if changed:
            self.redistribute_training_state()

    def redistribute_training_state(self):
        self.broadcast()
        self.repartition()

    def broadcast(self):
        for name, state in self._training_state.get_state().items():
            if hasattr(state, "__iter__"):
                for part in state:
                    part = self._broadcast_op(part)
            else:
                state = self._broadcast_op(state)

    def repartition(self):
        #  self._training_state._dataset.repartition()
        pass


class TrainingStateCallback(ms.train.callback.Callback):
    def __init__(self, training_state_manager, device, init=False):
        self._training_state_manager = training_state_manager
        self._init = init
        self._device = device

    def begin(self, run_context):
        if self._init:
            kfops.init(self._device)
        self._training_state_manager.broadcast()

    def step_begin(self, run_context):
        self._training_state_manager.check_for_change()

    def end(self, run_context):
        if self._init:
            kfops.finalize(self._device)
