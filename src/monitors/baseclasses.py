import numpy as np

class Monitor:
    def __init__(self, name, visualizing=False):
        self.value = None
        self.name = name
        self.scalar = None
        self.boolean = None
        self.visualizing = visualizing
        self.visualize_count = 0

    def initialize(self, statsdir, plotsdir):
        #print("INIT, {}, {}".format(savedir, self.name))
        self.plotsdir = plotsdir
        self.statsdir = statsdir

    def __call__(self, **kwargs):
        self.value = self.call(**kwargs)
        return self.value

    def visualize(self):
        if not self.visualizing:
            return
        self.visualize_count += 1

    def call(self, **kwargs):
        pass

    @property
    def string(self):
        return None

class ScalarMonitor(Monitor):
    def __init__(self, name, numerical=True,**kwargs):
        super().__init__(name, **kwargs)
        self.numerical = numerical
        self.scalar = True

    def visualize(self):
        super().visualize()

    @property
    def string(self):
        s = "\t{:>15s} = {:.2f}".format(self.name, self.value)
        return s
