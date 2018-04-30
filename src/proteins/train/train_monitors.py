from src.monitors import *
from ..protein_metrics import *
from src.admin.MonitorCollection import MonitorCollection


def get_monitor_collections(plotting_frequency):
    return dict(
        train=train_monitor_collection(plotting_frequency),
        valid=valid_monitor_collection(plotting_frequency),
        dummy_train=dummy_train_monitor_collection(plotting_frequency)
        )

def valid_monitor_collection(plotting_frequency):
    valid_loss = Collect('loss', ndp=3,visualizing=True)
    best_valid_loss = Best(valid_loss, track='min',ndp=3)
    protein_metrics = ProteinMetricCollection(
        'targets',
        'predictions',
        'masks',
        1,2,5,10,
        tracked_k=10,
        tracked_range='long',
        ndp=3,
        visualizing=True)
    metric_monitors = [
        protein_metrics,
        valid_loss,
        best_valid_loss,

    ]
    viz_monitors = [
        BatchMatrixMonitor('targets', plotting_frequency=plotting_frequency, batch_size=10, visualizing=True),
        BatchMatrixMonitor('half', plotting_frequency=plotting_frequency, batch_size=10, visualizing=True),
        BatchMatrixMonitor('hard_pred', plotting_frequency=plotting_frequency, batch_size=10, visualizing=True),
        BatchMatrixMonitor('predictions', plotting_frequency=plotting_frequency, batch_size=10, visualizing=True)
    ]
    monitors = metric_monitors + viz_monitors

    mc= MonitorCollection('valid', *monitors, plotting_frequency=plotting_frequency)

    mc.track_monitor = protein_metrics.track_monitor
    return mc

def train_monitor_collection(plotting_frequency):
    time_monitors = [
        Collect('epoch', visualizing=False,printing=False),
        Collect('iteration', visualizing=False, printing=False),
        Hours(),
        Collect('time', fn='sum', visualizing=False, printing=False),

    ]

    optim_monitors = [
        Collect('lr', fn='last', visualizing=True, ndp=8),
    ]



    monitors = optim_monitors + time_monitors + [Collect('loss', ndp=3,visualizing=True)]

    mc = MonitorCollection('train', *monitors, plotting_frequency=plotting_frequency)
    return mc

def dummy_train_monitor_collection(plotting_frequency):
    metric_monitors = [
        ProteinMetricCollection(
            'targets',
            'predictions',
            'masks',
            1,2,5,10,
            ndp=3,
            visualizing=True),
            ]
    monitors = metric_monitors
    mc = MonitorCollection('dummy_train', *monitors, plotting_frequency=plotting_frequency)
    return mc
