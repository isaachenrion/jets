
from collections import OrderedDict
class MonitorCollection:
    def __init__(self, **monitors):
        self.monitors = OrderedDict(**monitors)

    def initialize(self, *args, **kwargs):
        self._initialize_args = args
        self._initialize_kwargs = kwargs
        for m in self.monitors.values():
            m.initialize(*args, **kwargs)

    def __call__(self, **kwargs):
        return {name:m(**kwargs) for name, m in self.monitors.items()}

    def visualize(self, **kwargs):
        for m in self.monitors.values():
            m.visualize(**kwargs)

    def add_monitor(self, name, monitor):
        self.monitors[name] = monitor
        monitor.initialize(*self._initialize_args, **self._initialize_kwargs)
        return monitor

    def add_monitors(self, **monitors):
        for name, monitor in monitors.items():
            self.add_monitor(name, monitor)

    @property
    def visualized_scalar_monitor_names(self):
        return [m.name for m in self.monitors.values() if m.scalar and m.visualizing]

    @property
    def scalar_monitor_names(self):
        return [m.name for m in self.monitors.values() if m.scalar]

    @property
    def string(self):
        s = "\n".join([m.string for _, m in self.monitors.items() if m.string is not None])
        return s
