import numpy as np

class Monitor:
    def __init__(self, name, visualizing=False, printing=True):
        self.value = None
        self.name = name
        self.scalar = None
        self.boolean = None
        self.visualizing = visualizing
        self.printing = printing
        self.visualize_count = 0
        self.call_count = 0
        self.plotsdir = None
        self.statsdir = None
        self.epoch = 0
        self.batch = 0

    def initialize(self, statsdir, plotsdir):
        #print("INIT, {}, {}".format(savedir, self.name))
        self.plotsdir = plotsdir
        self.statsdir = statsdir

    def new_epoch(self):
        self.epoch += 1
        self.batch = 0

    def __call__(self, **kwargs):
        self.value = self.call(**kwargs)
        self.batch += 1
        self.call_count += 1
        return self.value

    def visualize(self):
        if not self.visualizing:
            return
        else:
            self._visualize()
            self.visualize_count += 1

    def _visualize(self):
        raise NotImplementedError

    @property
    def string(self):
        if not self.printing:
            return None
        else:
            return self._string

    @property
    def _string(self):
        return None

    def call(self, **kwargs):
        pass


class ScalarMonitor(Monitor):
    def __init__(self, name, numerical=True, ndp=2,**kwargs):
        super().__init__(name, **kwargs)
        self.numerical = numerical
        self.scalar = True
        self.ndp = ndp

    @property
    def _string(self):
        if self.numerical:
            s = "\t{:>15s} = {:.{x}f}".format(self.name, self.value, x=self.ndp)
        else:
            s = "\t{} = {}".format(self.name, self.value)

        return s
