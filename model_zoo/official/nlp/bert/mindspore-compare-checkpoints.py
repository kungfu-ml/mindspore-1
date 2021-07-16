#!/usr/bin/env python3.7
import sys
from hashlib import sha1

import mindspore as ms
import numpy as np
from libtensor_diff import tensor_diff_f32


def show_tensor_diff(name, width, x, y):
    title = '%-*s  %-12s %-24s  |  ' % (width, name, x.dtype, x.shape)
    if x.dtype != np.float32:
        print(title + 'only support float32')
        return (0, True)  # Assuming int tensors are equal
    if y.dtype != np.float32:
        print(title + 'only support float32')
        return (0, True)  # Assuming int tensors are equal
    r = tensor_diff_f32(x, y)
    print(title + 'max_diff: %f, bytes_equal: %s' %
          (r.max_diff, r.bytes_equal))
    return (r.max_diff, r.bytes_equal)


def compare_checkpoints(f1, f2):
    params1 = ms.train.serialization.load_checkpoint(f1)
    params2 = ms.train.serialization.load_checkpoint(f2)

    kvs1 = dict((k, p.asnumpy()) for k, p in params1.items())
    kvs2 = dict((k, p.asnumpy()) for k, p in params2.items())

    names = list(sorted(set(list(kvs1.keys()) + list(kvs2.keys()))))

    width = max(len(name) for name in names)

    differences = []

    for name in names:
        x = kvs1.get(name)
        y = kvs2.get(name)

        if x is None:
            print('missing {} from {}'.format(name, f1))
            continue

        if y is None:
            print('missing {} from {}'.format(name, f2))
            continue

        diff, eq = show_tensor_diff(name, width, x, y)
        differences += [(diff, eq, name)]

    print('\n\n')
    print('sort by diff')
    for d, eq, name in reversed(sorted(differences)):
        if not eq:
            print('%12f  %6s  %s' % (d, eq, name))

    # for k, p in params1.items():
    #     x = p.asnumpy()
    # h = sha1(x.tobytes()).hexdigest()
    # meta = '%s%-20s' % (x.dtype, x.shape)
    # stat = '[%f, %f] ~ %f' % (x.min(), x.max(), x.mean())
    # print('[%3d]    %s    %-24s: %s %s' % (idx, h, k, meta, stat))


def main(args):
    if len(args) != 2:
        print('Usage: <f1> <f2>')
    compare_checkpoints(args[0], args[1])


main(sys.argv[1:])
