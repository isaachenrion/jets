
from collections import OrderedDict
class MonitorCollection:
    def __init__(self, name, *monitors, plotting_frequency=1, track_monitor=None,**kw_monitors):
        self.name = name
        self.monitors = OrderedDict()
        for m in monitors:
            self.monitors[m.name] = m
        self.monitors.update(kw_monitors)
        self.track_monitor = track_monitor
        self.visualize_calls = 0
        self.plotting_frequency = plotting_frequency
        self.subcollections = None

    @property
    def monitor_names(self):
        return list(self.monitors.keys())

    @property
    def visualized_scalar_monitor_names(self):
        out = [m.name for m in self.monitors.values() if m.scalar and m.visualizing]
        if self.subcollections is not None:
            for c in self.subcollections:
                out += c.visualized_scalar_monitor_names
        return out

    @property
    def scalar_monitor_names(self):
        out = [m.name for m in self.monitors.values() if m.scalar]
        if self.subcollections is not None:
            for c in self.subcollections:
                out += c.scalar_monitor_names
        return out

    def initialize(self, *args, **kwargs):
        self._initialize_args = args
        self._initialize_kwargs = kwargs
        for m in self.monitors.values():
            m.initialize(*args, **kwargs)

        if self.subcollections is not None:
            for c in self.subcollections:
                c.initialize(*args, **kwargs)


    def __call__(self, **kwargs):
        out_dict = {name:self.monitors[name](**kwargs) for name in self.monitor_names}
        if self.subcollections is not None:
            for c in self.subcollections:
                out_dict.update(c(**kwargs))
        return out_dict

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
            monitor.initialize(*self._initialize_args, **self._initialize_kwargs)
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

    def add_subcollection(self, collection):
        if self.subcollections is None:
            self.subcollections = [collection]
        else:
            self.subcollections.append(collection)
