from src.monitors import *
from ..protein_metrics import *
from src.admin.MonitorCollection import MonitorCollection


def get_monitor_collections(logging_frequency):
    return dict(
        train=train_monitor_collection(),
        valid=valid_monitor_collection(logging_frequency),
        dummy_train=dummy_train_monitor_collection()
        )

def valid_monitor_collection(logging_frequency):
    valid_loss = Regurgitate('loss', ndp=3,visualizing=True)
    best_valid_loss = Best(valid_loss, track='min',ndp=3)
    metric_monitors = [
        ProteinMetricCollection(
            'targets',
            'predictions',
            'masks',
            1,2,5,10,
            ndp=3,
            visualizing=True),
        valid_loss,
        best_valid_loss,

    ]
    viz_monitors = [
        BatchMatrixMonitor('targets', n_epochs=logging_frequency, batch_size=10, visualizing=True),
        BatchMatrixMonitor('half', n_epochs=logging_frequency, batch_size=10, visualizing=True),
        BatchMatrixMonitor('hard_pred', n_epochs=logging_frequency, batch_size=10, visualizing=True),
        BatchMatrixMonitor('predictions', n_epochs=logging_frequency, batch_size=10, visualizing=True)
    ]
    monitors = metric_monitors + viz_monitors

    mc= MonitorCollection('valid', *monitors)

    mc.track_monitor = best_valid_loss
    return mc

def train_monitor_collection():
    time_monitors = [
        Regurgitate('epoch', visualizing=False,printing=False),
        Regurgitate('iteration', visualizing=False, printing=False),
        Hours(),
        Collect('time', fn='sum', visualizing=False, printing=False),

    ]

    optim_monitors = [
        Collect('lr', fn='last', visualizing=True, ndp=8),
    ]



    monitors = optim_monitors + time_monitors + [Regurgitate('loss', ndp=3,visualizing=True)]

    mc = MonitorCollection('train', *monitors)
    return mc

def dummy_train_monitor_collection():
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
    mc = MonitorCollection('dummy_train', *monitors)
    return mc
