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
        self.all_reduce = KungFuAllReduce()

        # FIXME: make it dynamic
        cluster_size = float(kfops.kungfu_current_cluster_size())
        scalar_shape = []
        self.cluster_size = ms.Parameter(
            ms.Tensor(
                np.ones(scalar_shape) * cluster_size,
                ms.int32,
            ))

        self._rank = kfops.kungfu_current_rank()

    def construct(self, gradients):
        gradients = self.map_(self.all_reduce, gradients)

        # debug
        #  for i, grad in enumerate(gradients):
            #  np.save("./grads/grad-{}-{}.npy".format(self._rank, i), grad.asnumpy())

        mean_grads = self.hyper_map(F.partial(grad_scale, self.cluster_size),
                                    gradients)

        # debug
        for i, grad in enumerate(gradients):
            np.save("./grads/grad-{}-{}.npy".format(self._rank, i), grad.asnumpy())

        return super(KungFuLambDebug, self).construct(mean_grads)


class KungFuLambDebugModel(ms.nn.Lamb):
    def __init__(self, *args, **kwargs):
        super(KungFuLambDebugModel, self).__init__(*args, **kwargs)
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

        self._rank = kfops.kungfu_current_rank()

    def construct(self, gradients):
        gradients = self.map_(self.all_reduce, gradients)
        mean_grads = self.hyper_map(F.partial(grad_scale, self.cluster_size),
                                    gradients)

        x = super(KungFuLambDebugModel, self).construct(mean_grads)

        # debug
        for i, param in enumerate(self.params):
            np.save("./params/param-{}-{}.npy".format(self._rank, i), param.asnumpy())

        return x


class LambDebug(ms.nn.Lamb):
    def __init__(self, *args, **kwargs):
        super(LambDebug, self).__init__(*args, **kwargs)
        self._rank = 0

    def construct(self, gradients):
        for i, grad in enumerate(gradients):
            np.save("./grads/grad-{}-{}.npy".format(self._rank, i), grad.asnumpy())

        return super(LambDebug, self).construct(gradients)
