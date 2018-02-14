import numpy as np

class Monitor:
    def __init__(self, name, visualizing=False):
        self.value = None
        self.name = name
        self.scalar = None
        self.boolean = None
        self.visualizing = visualizing

    def initialize(self, statsdir, plotsdir):
        #print("INIT, {}, {}".format(savedir, self.name))
        self.plotsdir = plotsdir
        self.statsdir = statsdir

    def __call__(self, **kwargs):
        self.value = self.call(**kwargs)
        if self.visualizing:
            self.visualize()
        return self.value

    def visualize(self):
        pass

    def call(self, **kwargs):
        pass

    def finish(self):
        pass

class ScalarMonitor(Monitor):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.scalar = True

    def visualize(self):
        pass
