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
        self.visualize_count += 1

    def call(self, **kwargs):
        pass

    def finish(self):
        pass

class ScalarMonitor(Monitor):
    def __init__(self, name, numerical=True,**kwargs):
        super().__init__(name, **kwargs)
        self.numerical = numerical
        self.scalar = True

    def visualize(self):
        super().visualize()

        #self.visualize_count += 1

    @property
    def string(self):
        return "\t{} = {:.2f}\t".format(self.name, self.value)
