from src.monitors import *
from src.admin.MonitorCollection import MonitorCollection

def train_monitor_collection(logging_frequency):
    valid_loss = Regurgitate('valid_loss', ndp=3,visualizing=True)
    best_valid_loss = Best(valid_loss, track='min',ndp=3)
    metric_monitors = [
        ProteinMetricCollection(1,2,5,10,ndp=3,visualizing=True),
        valid_loss,
        best_valid_loss,
        Regurgitate('train_loss', ndp=3,visualizing=True)

    ]
    #self.metric_monitors = metric_monitors

    time_monitors = [
        Regurgitate('epoch', visualizing=False,printing=False),
        Regurgitate('iteration', visualizing=False, printing=False),
        Hours(),
        Collect('time', fn='sum', visualizing=False, printing=False),

    ]

    optim_monitors = [
        Collect('lr', fn='last', visualizing=True, ndp=8),
    ]


    viz_monitors = [
        BatchMatrixMonitor('yy', n_epochs=logging_frequency, batch_size=10, visualizing=True),
        BatchMatrixMonitor('half', n_epochs=logging_frequency, batch_size=10, visualizing=True),
        BatchMatrixMonitor('hard_pred', n_epochs=logging_frequency, batch_size=10, visualizing=True),
        BatchMatrixMonitor('yy_pred', n_epochs=logging_frequency, batch_size=10, visualizing=True)
    ]

    monitors = metric_monitors + optim_monitors + time_monitors
    monitors += viz_monitors

    mc = MonitorCollection(*monitors)
    mc.track_monitor = best_valid_loss
    return mc

def test_monitor_collection():
    test_loss = Regurgitate('test_loss', visualizing=True)
    #best_test_loss = Best(test_loss)

    monitors = [
        ProteinMetrics(k=1,visualizing=True),
        ProteinMetrics(k=2,visualizing=True),
        ProteinMetrics(k=5,visualizing=True),
        ProteinMetrics(k=10,visualizing=True),
        test_loss,
        #best_test_loss,
    ]
    mc = MonitorCollection(*monitors)

    return mc
