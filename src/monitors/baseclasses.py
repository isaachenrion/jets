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

    def initialize(self, statsdir, plotsdir):
        #print("INIT, {}, {}".format(savedir, self.name))
        self.plotsdir = plotsdir
        self.statsdir = statsdir

    def __call__(self, **kwargs):
        self.call_count += 1
        self.value = self.call(**kwargs)
        return self.value

    def visualize(self):
        if not self.visualizing:
            return
        self.visualize_count += 1

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

    def visualize(self):
        super().visualize()

    @property
    def _string(self):
        if self.numerical:
            s = "\t{:>15s} = {:.{x}f}".format(self.name, self.value, x=self.ndp)
        else:
            s = "\t{} = {}".format(self.name, self.value)

        return s
