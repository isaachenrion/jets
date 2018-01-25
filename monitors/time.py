import datetime
from .baseclasses import ScalarMonitor

class Hours(ScalarMonitor):
    def __init__(self, **kwargs):
        super().__init__('hours', **kwargs)

    def call(self, time=None, **kwargs):
        self.value = time / 3600
        return self.value


class ETA(ScalarMonitor):
    def __init__(self, start_time, epochs, **kwargs):
        super().__init__('eta', **kwargs)
        self.start_time = start_time
        self.epochs = epochs
        self.on_epoch = 0
        #self.epoch_collector = Collect('time', fn='mean', visualizing=False)

    def call(self, time=None, **kwargs):
        self.on_epoch += 1
        mean_epoch_time = time/self.on_epoch
        estimated_total_time = mean_epoch_time * self.epochs
        self.value = self.start_time + datetime.timedelta(0, estimated_total_time)
        self.value = self.value.strftime('%c')
        #import ipdb; ipdb.set_trace()
        return self.value
