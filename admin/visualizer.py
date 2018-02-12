import visdom
import numpy as np

class Visualizer:
    def __init__(self):
        self.viz = visdom.Visdom()

    def histogram(self, data, n_bins=30):
        self.viz.histogram(X=data, opts=dict(numbins=n_bins))

    def line(self, y):
        self.viz.line(Y=y, opts=dict(showlegend=True))
