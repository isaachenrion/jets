import numpy as np

class Monitor:
    def __init__(self, name, visualizing=True):
        self.value = None
        self.name = name
        self.scalar = None
        self.boolean = None
        self.visualizing = visualizing

    def initialize(self, statsdir, plotsdir, viz):
        #print("INIT, {}, {}".format(savedir, self.name))
        self.plotsdir = plotsdir
        self.statsdir = statsdir
        if self.visualizing:
            self.viz = viz
        else:
            self.viz = None

    def __call__(self, **kwargs):
        self.value = self.call(**kwargs)
        if self.visualizing and self.viz is not None:
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
        #self.boolean = boolean
        self.counter = 0

    def visualize(self):
        self.counter += 1
        if self.counter == 1:
            self.viz.line(
                X=np.array((self.counter,)),
                Y=np.array((self.value,)),
                win=self.name,
                opts=dict(
                    fillarea=False,
                    showlegend=True,
                    width=400,
                    height=400,
                    xlabel='Epochs',
                    ylabel=self.name,
                    title=self.name,
                    marginleft=30,
                    marginright=30,
                    marginbottom=80,
                    margintop=30,
                )
            )
        else:
            self.viz.line(
                X=np.array((self.counter,)),
                Y=np.array((self.value,)),
                win=self.name,
                update='append'
            )
