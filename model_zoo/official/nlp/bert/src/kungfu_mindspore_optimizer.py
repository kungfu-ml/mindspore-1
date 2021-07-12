import mindspore as ms
import mindspore.ops.operations.kungfu_comm_ops as kfops
import numpy as np
from mindspore.common import dtype as mstype
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.operations.kungfu_comm_ops import KungFuAllReduce

cast = P.Cast()

grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(cast(scale, mstype.float32))


class KungFuMomentum(ms.nn.Momentum):
    def __init__(self, *args, **kwargs):
        super(KungFuMomentum, self).__init__(*args, **kwargs)
        self.map_ = C.Map()
        self.all_reduce = KungFuAllReduce()

        # FIXME: make it dynamic
        cluster_size = float(kfops.kungfu_current_cluster_size())
        scalar_shape = []
        self.cluster_size = ms.Parameter(
            ms.Tensor(
                np.ones(scalar_shape) * cluster_size,
                ms.int32,
            ))

    def construct(self, gradients):
        gradients = self.map_(self.all_reduce, gradients)
        mean_grads = self.hyper_map(F.partial(grad_scale, self.cluster_size),
                                    gradients)
        return super(KungFuMomentum, self).construct(mean_grads)


class KungFuLamb(ms.nn.Lamb):
    def __init__(self, *args, **kwargs):
        super(KungFuLamb, self).__init__(*args, **kwargs)
        self.map_ = C.Map()
        self.all_reduce = KungFuAllReduce()

        # FIXME: make it dynamic
        cluster_size = float(kfops.kungfu_current_cluster_size())
        scalar_shape = []
        self.cluster_size = ms.Parameter(
            ms.Tensor(
                np.ones(scalar_shape) * cluster_size,
                ms.int32,
            ))

    def construct(self, gradients):
        gradients = self.map_(self.all_reduce, gradients)
        mean_grads = self.hyper_map(F.partial(grad_scale, self.cluster_size),
                                    gradients)
        return super(KungFuLamb, self).construct(mean_grads)


class KungFuLambDebug(ms.nn.Lamb):
    def __init__(self, *args, **kwargs):
        super(KungFuLambDebug, self).__init__(*args, **kwargs)
        self.map_ = C.Map()
        self.log_tensor = kfops.KungFuLogTensor()

    def construct(self, gradients):
        print("Type: {}".format(type(gradients[0])))
        for i, grad in enumerate(gradients):
            np.save("{}.npy".format(i), grad.asnumpy())

        gradients = self.map_(self.log_tensor, gradients)
        return super(KungFuLambDebug, self).construct(gradients)
