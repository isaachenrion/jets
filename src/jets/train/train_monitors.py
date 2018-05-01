from src.monitors import *
from src.admin.MonitorCollection import MonitorCollection


def get_monitor_collections(plotting_frequency):
    return dict(
        train=train_monitor_collection(plotting_frequency),
        valid=valid_monitor_collection(plotting_frequency),
        dummy_train=dummy_train_monitor_collection(plotting_frequency)
        )

def train_monitor_collection(plotting_frequency):
    time_monitors = [
        Collect('epoch', visualizing=False, printing=False),
        Collect('iteration', visualizing=False, printing=False),
        Hours(),
    ]
    optim_monitors = [
        Collect('lr', fn='last', ndp=8,visualizing=True),
    ]
    monitors = time_monitors + optim_monitors + [Collect('loss', ndp=3,visualizing=True)]
    mc = MonitorCollection('train',*monitors, plotting_frequency=plotting_frequency)
    return mc

def valid_monitor_collection(plotting_frequency):
    roc_auc = ROCAUC(visualizing=True, ndp=5)
    inv_fpr = InvFPR(visualizing=True)
    #best_inv_fpr = Best(inv_fpr)
    #roc_auc_at_best_inv_fpr = LogOnImprovement(roc_auc, best_inv_fpr)

    valid_loss = Collect('loss', ndp=3,visualizing=True)
    best_valid_loss = Best(valid_loss)
    inv_fpr_at_best_valid_loss = LogOnImprovement(inv_fpr, best_valid_loss)
    roc_auc_at_best_valid_loss = LogOnImprovement(roc_auc, best_valid_loss)

    metric_monitors = [
        inv_fpr,
        #best_inv_fpr,
        roc_auc,
        #roc_auc_at_best_inv_fpr,
        valid_loss,
        best_valid_loss,
        inv_fpr_at_best_valid_loss,
        roc_auc_at_best_valid_loss

    ]

    #grad_monitors = [
    #    GradNorm(visualizing=True),
    #    ParamNorm( visualizing=True),
    #    UpdateRatio( visualizing=True)
    #]

    monitors = metric_monitors  #+ grad_monitors
    #monitors += viz_monitors

    mc = MonitorCollection('valid',*monitors, plotting_frequency=plotting_frequency)
    mc.track_monitor = best_valid_loss
    return mc


def dummy_train_monitor_collection(plotting_frequency):
    roc_auc = ROCAUC(visualizing=True, ndp=5)
    inv_fpr = InvFPR(visualizing=True)
    #best_inv_fpr = Best(inv_fpr)
    #roc_auc_at_best_inv_fpr = LogOnImprovement(roc_auc, best_inv_fpr)

    valid_loss = Collect('loss', ndp=3,visualizing=True)
    best_valid_loss = Best(valid_loss)
    inv_fpr_at_best_valid_loss = LogOnImprovement(inv_fpr, best_valid_loss)
    roc_auc_at_best_valid_loss = LogOnImprovement(roc_auc, best_valid_loss)

    metric_monitors = [
        inv_fpr,
        #best_inv_fpr,
        roc_auc,
        #roc_auc_at_best_inv_fpr,
        valid_loss,
        best_valid_loss,
        inv_fpr_at_best_valid_loss,
        roc_auc_at_best_valid_loss

    ]


    monitors = metric_monitors  #

    mc = MonitorCollection('dummy_train',*monitors, plotting_frequency=plotting_frequency)
    return mc
