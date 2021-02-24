# Copyright 2021 Huawei Technologies Co., Ltd
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
""" test scatter update """
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor, Model, Parameter
from mindspore.ops import operations as P
from mindspore import context


class Net(nn.Cell):
    """Net definition"""
    def __init__(self):
        super(Net, self).__init__()
        self.inputs = Parameter(Tensor(np.ones([32, 128]).astype(np.float32)), "input")
        self.indices = Tensor(np.ones([4]).astype(np.int32))
        self.updates = Tensor(np.ones([4, 128]).astype(np.float32))
        self.scatter_update = P.ScatterUpdate().shard(((1, 8), (1,), (1, 8)))
        self.add = P.TensorAdd().shard(((8, 1), (8, 1)))
        self.relu = P.ReLU()

    def construct(self, x):
        out = self.scatter_update(self.inputs, self.indices, self.updates)
        out = self.add(x, out)
        out = self.relu(out)
        return out


def test_distribute_predict():
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, full_batch=True)
    inputs = Tensor(np.ones([32, 128]).astype(np.float32))
    net = Net()
    model = Model(net)
    predict_map = model.infer_predict_layout(inputs)
    output = model.predict(inputs)
    context.reset_auto_parallel_context()
    return predict_map, output
