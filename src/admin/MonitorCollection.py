
from collections import OrderedDict
class MonitorCollection:
    def __init__(self, name, *monitors):
        self.name = name
        self.monitors = OrderedDict()
        for m in monitors:
            self.monitors[m.name] = m
        self.track_monitor = None

    @property
    def monitor_names(self):
        return list(self.monitors.keys())

    @property
    def visualized_scalar_monitor_names(self):
        return [m.name for m in self.monitors.values() if m.scalar and m.visualizing]

    @property
    def scalar_monitor_names(self):
        return [m.name for m in self.monitors.values() if m.scalar]

    def initialize(self, *args, **kwargs):
        self._initialize_args = args
        self._initialize_kwargs = kwargs
        for m in self.monitors.values():
            m.initialize(*args, **kwargs)

    def __call__(self, **kwargs):
        return {name:self.monitors[name](**kwargs) for name in self.monitor_names}

    def visualize(self, **kwargs):
        for m in self.monitors.values():
            m.visualize(**kwargs)

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
        return s
