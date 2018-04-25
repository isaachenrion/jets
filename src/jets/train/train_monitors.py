from src.monitors import *
from src.admin.MonitorCollection import MonitorCollection


def get_monitor_collections(logging_frequency):
    return dict(
        train=train_monitor_collection(logging_frequency),
        valid=valid_monitor_collection(logging_frequency),
        dummy_train=dummy_train_monitor_collection(logging_frequency)
        )

def train_monitor_collection(logging_frequency):
    time_monitors = [
        Collect('epoch', visualizing=False, printing=False),
        Collect('iteration', visualizing=False, printing=False),
        Hours(),
    ]
    optim_monitors = [
        Collect('lr', fn='last', ndp=8,visualizing=True),
    ]
    monitors = time_monitors + optim_monitors + [Collect('loss', ndp=3,visualizing=True)]
    mc = MonitorCollection('train',*monitors)
    return mc

def valid_monitor_collection(logging_frequency):
    roc_auc = ROCAUC(visualizing=True, ndp=5)
    inv_fpr = InvFPR(visualizing=True)
    best_inv_fpr = Best(inv_fpr)
    roc_auc_at_best_inv_fpr = LogOnImprovement(roc_auc, best_inv_fpr)

    metric_monitors = [
        inv_fpr,
        best_inv_fpr,
        roc_auc,
        roc_auc_at_best_inv_fpr,
        Collect('loss', ndp=3,visualizing=True),
    ]

    #grad_monitors = [
    #    GradNorm(visualizing=True),
    #    ParamNorm( visualizing=True),
    #    UpdateRatio( visualizing=True)
    #]

    monitors = metric_monitors  #+ grad_monitors
    #monitors += viz_monitors

    mc = MonitorCollection('valid',*monitors)
    mc.track_monitor = best_inv_fpr
    return mc


def dummy_train_monitor_collection(logging_frequency):
    roc_auc = ROCAUC(visualizing=True, ndp=5)
    inv_fpr = InvFPR(visualizing=True)
    best_inv_fpr = Best(inv_fpr)
    roc_auc_at_best_inv_fpr = LogOnImprovement(roc_auc, best_inv_fpr)

    metric_monitors = [
        inv_fpr,
        best_inv_fpr,
        roc_auc,
        roc_auc_at_best_inv_fpr,
        Collect('loss', ndp=3,visualizing=True),

    ]


    monitors = metric_monitors  #

    mc = MonitorCollection('dummy_train',*monitors)
    return mc
