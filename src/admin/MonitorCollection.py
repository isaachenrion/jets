import os
from collections import OrderedDict


class MonitorCollection:
    def __init__(self, name, *monitors, plotting_frequency=1, track_monitor=None,**kw_monitors):
        self.name = name
        self._monitors = OrderedDict()
        for m in monitors:
            self._monitors[m.name] = m
        self._monitors.update(kw_monitors)
        self.track_monitor = track_monitor
        self.visualize_calls = 0
        self.plotting_frequency = plotting_frequency
        self.subcollections = None

    @property
    def names(self):
        l = list(self.monitors.keys())
        if self.subcollections is not None:
            for c in self.subcollections:
                l += c.names
        return l

    @property
    def monitors(self):
        return self._monitors


    @property
    def visualized_scalar_names(self):
        out = [m.name for m in self.monitors.values() if m.scalar and m.visualizing]
        if self.subcollections is not None:
            for c in self.subcollections:
                out += c.visualized_scalar_names
        return out

    @property
    def scalar_names(self):
        out = [m.name for m in self.monitors.values() if m.scalar]
        if self.subcollections is not None:
            for c in self.subcollections:
                out += c.scalar_names
        return out

    def initialize(self, statsdir, plotsdir):
        self.statsdir = statsdir
        self.plotsdir = plotsdir
        if not os.path.exists(self.statsdir):
            os.makedirs(self.statsdir)
        if not os.path.exists(self.plotsdir):
            os.makedirs(self.plotsdir)

        for m in self.monitors.values():
            m.initialize(statsdir, plotsdir)

        if self.subcollections is not None:
            for c in self.subcollections:
                c.initialize(statsdir, plotsdir)


    def __call__(self, **kwargs):
        out_dict = {name:monitor(**kwargs) for name, monitor in self.monitors.items()}
        if self.subcollections is not None:
            for c in self.subcollections:
                out_dict.update(c(**kwargs))
        return out_dict

    @property
    def scalar_values(self):
        d = {n:m.value for n, m in self.monitors.items() if m.scalar}
        if self.subcollections is not None:
            for c in self.subcollections:
                d.update(c.scalar_values)
        return d

    def visualize(self, **kwargs):
        if self.visualize_calls % self.plotting_frequency == 0:
            for m in self.monitors.values():
                m.visualize(**kwargs)
        if self.subcollections is not None:
            for c in self.subcollections:
                c.visualize(**kwargs)

        self.visualize_calls += 1


    def add_monitor(self, monitor, initialize=False):
        self.monitors[monitor.name] = monitor
        if initialize:
            monitor.initialize(self.statsdir, self.plotsdir)
        return monitor

    def add_monitors(self, *monitors, initialize=False):
        for monitor in monitors:
            self.add_monitor(monitor, initialize)

    @property
    def string(self):
        s = ''
        s += '\n{} monitors\n'.format(self.name)
        s += "\n".join([m.string for _, m in self.monitors.items() if m.string is not None])
        if self.subcollections is not None:
            for c in self.subcollections:
                s += c.string
        return s


    def new_epoch(self):
        for monitor in self.monitors.values():
            monitor.new_epoch()
        if self.subcollections is not None:
            for c in self.subcollections:
                c.new_epoch()

    def add_subcollection(self, collection):
        if self.subcollections is None:
            self.subcollections = [collection]
        else:
            self.subcollections.append(collection)
