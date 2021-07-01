import mindspore as ms
from mindspore.ops import composite as C
from mindspore.ops.operations.kungfu_comm_ops import KungFuAllReduce


class KungFuMomentum(ms.nn.Momentum):
    def __init__(self, *args, **kwargs):
        super(KungFuMomentum, self).__init__(*args, **kwargs)
        self.map_ = C.Map()
        self.all_reduce = KungFuAllReduce()

    def construct(self, gradients):
        gradients = self.map_(self.all_reduce, gradients)
        return super(KungFuMomentum, self).construct(gradients)


class KungFuLamb(ms.nn.Lamb):
    def __init__(self, *args, **kwargs):
        super(KungFuLamb, self).__init__(*args, **kwargs)
        self.map_ = C.Map()
        self.all_reduce = KungFuAllReduce()

    def construct(self, gradients):
        gradients = self.map_(self.all_reduce, gradients)
        return super(KungFuLamb, self).construct(gradients)
