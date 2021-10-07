import os
import sys
import time

import mindspore as ms
from kungfu._utils import show_duration
from kungfu.python.elastic_state import ElasticContext, ElasticState

__all__ = [
    'ElasticState',
    'ElasticContext',
    'ElasticCallback',
]


def estimate_remain(p, d):
    if p == 0:
        return 1e10
    return (1 - p) * d / p


class ElasticCallback(ms.train.callback.Callback):
    def __init__(self, elastic_state, global_batch_size):
        self._elastic_state = elastic_state
        self._global_batch_size = global_batch_size
        self._job_start = int(os.getenv('KUNGFU_JOB_START_TIMESTAMP'))

    def begin(self, run_context):
        pass
        print('ElasticCallback::begin')

    def epoch_begin(self, run_context):
        pass
        print('ElasticCallback::epoch_begin')

    def epoch_end(self, run_context):
        pass
        print('ElasticCallback::epoch_end')

    def step_begin(self, run_context):
        print('ElasticCallback::step_begin')
        should_sync = self._elastic_state.begin()
        if should_sync:
            print(
                'TODO: sync state to %d, no need to sync dataset state in reload mode'
                % (self._elastic_state._progress))

        duration = time.time() - self._job_start
        p = (float(self._elastic_state._progress) /
             float(self._elastic_state._max_progress))

        print('progress: %d/%d, took %s, remain: %s' % (
            self._elastic_state._progress,
            self._elastic_state._max_progress,
            show_duration(duration),
            show_duration(estimate_remain(p, duration)),
        ))

    def step_end(self, run_context):
        print('ElasticCallback::step_end')
        self._elastic_state.end(self._global_batch_size)
        if self._elastic_state.stopped():
            print('_elastic_state stopped, requesting run_context to stop')
            run_context.request_stop()

            d = self._elastic_state.get_duration_since_resize()
            print('from resize start to after request stop: %dms' % (d / 1e6))

    def end(self, run_context):
        pass
        print('StopCallback::end')
