import torch as th
import time

class Timer:
    '''
        timer class with with statement in milliseconds
        usage:
        with Timer('cuda'|'cpu') as t:
            # code block
        time = t.result
    '''
    def __init__(self, device:str='cpu'):
        self.device = str(device)
        if self.device != 'cpu':
            self.start_event = th.cuda.Event(enable_timing=True)
            self.end_event = th.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.device == 'cpu':
            self.start_time = time.time()
        else:
            self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.device == 'cpu':
            self.end_time = time.time()
            self.result = (self.end_time - self.start_time) * 1000
        else:
            self.end_event.record()
            th.cuda.synchronize()
            self.result = self.start_event.elapsed_time(self.end_event)
    