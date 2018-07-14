from src.monitors import *
from ..protein_metrics import *
from src.admin.MonitorCollection import MonitorCollection


def get_monitor_collections(plotting_frequency):
    return dict(
        train=train_monitor_collection(plotting_frequency),
        valid=valid_monitor_collection(plotting_frequency),
        dummy_train=valid_monitor_collection(plotting_frequency)
        )

def valid_monitor_collection(plotting_frequency):
    valid_loss = Collect('loss', ndp=3,visualizing=False)
    best_valid_loss = Best(valid_loss, track='min',ndp=3)
    protein_metrics = ProteinMetricCollection(
        'dists',
        'pred',
        'cmask',
        1,2,5,10,
        tracked_k=10,
        tracked_range='long',
        ndp=3,
        visualizing=False)
    metric_monitors = [
        #protein_metrics,
        valid_loss,
        best_valid_loss,

    ]
    coords_cm = ContactMapMonitor('coords', mask_name='batch_mask', data_type='coords', plotting_frequency=plotting_frequency, batch_size=10, visualizing=True)
    preds_cm = ContactMapMonitor('pred', mask_name='batch_mask', data_type='logits', plotting_frequency=plotting_frequency, batch_size=10, visualizing=True)

    viz_monitors = [
        coords_cm,
        preds_cm,
        SplitBatchMatrixMonitor(coords_cm, preds_cm, 'ContactMap-split',plotting_frequency=plotting_frequency, batch_size=10, visualizing=True)
        #BatchMatrixMonitor('hard_pred', plotting_frequency=plotting_frequency, batch_size=10, visualizing=True),
        #BatchMatrixMonitor('predictions', plotting_frequency=plotting_frequency, batch_size=10, visualizing=True)
    ]
    monitors = metric_monitors + viz_monitors

    mc= MonitorCollection('valid', *monitors, plotting_frequency=plotting_frequency)


    mc.add_subcollection(protein_metrics)
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



    monitors = optim_monitors + time_monitors + [Collect('loss', ndp=3,visualizing=False)]

    mc = MonitorCollection('train', *monitors, plotting_frequency=plotting_frequency)
    return mc
