
from collections import OrderedDict
class MonitorCollection:
    def __init__(self, *monitors):
        self.monitor_names = [m.name for m in monitors]
        self.monitors = OrderedDict()
        for i, name in enumerate(self.monitor_names):
            self.monitors[name] = monitors[i]
        self.track_monitor = None

    @property
    def visualized_scalar_monitor_names(self):
        try:
            return self._visualized_scalar_monitor_names
        except AttributeError:
            self._visualized_scalar_monitor_names = [m.name for m in self.monitors.values() if m.scalar and m.visualizing]
            return self._visualized_scalar_monitor_names

    @property
    def scalar_monitor_names(self):
        try:
            return self._scalar_monitor_names
        except AttributeError:
            self._scalar_monitor_names = [m.name for m in self.monitors.values() if m.scalar]
            return self._scalar_monitor_names

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
        s = "\n".join([m.string for _, m in self.monitors.items() if m.string is not None])
        return s
