from src.monitors import *
from src.admin.MonitorCollection import MonitorCollection

def get_monitor_collections():
    return dict(test=test_monitor_collection())

def ODLtest_monitor_collection():
    roc_auc = ROCAUC(visualizing=True)
    inv_fpr = InvFPR(visualizing=True)
    best_inv_fpr = Best(inv_fpr)

    monitors = [
        inv_fpr,
        best_inv_fpr,
        roc_auc,
        Collect('loss', visualizing=True),
        Collect('model', visualizing=False, numerical=False)
        ]
    mc = MonitorCollection('test', *monitors)

    return mc

    return mc

def test_monitor_collection():
    roc_auc = ROCAUC(visualizing=False, ndp=5)
    inv_fpr = InvFPR(visualizing=False)
    best_inv_fpr = Best(inv_fpr)
    #roc_auc_at_best_inv_fpr = LogOnImprovement(roc_auc, best_inv_fpr)

    valid_loss = Collect('loss', ndp=3,visualizing=False)
    best_valid_loss = Best(valid_loss, track='min')
    inv_fpr_at_best_valid_loss = LogOnImprovement(inv_fpr, best_valid_loss)
    roc_auc_at_best_valid_loss = LogOnImprovement(roc_auc, best_valid_loss)

    metric_monitors = [
        inv_fpr,
        best_inv_fpr,
        roc_auc,
        #roc_auc_at_best_inv_fpr,
        valid_loss,
        best_valid_loss,
        inv_fpr_at_best_valid_loss,
        roc_auc_at_best_valid_loss,
        Collect('model_name', visualizing=False, numerical=False)
    ]

    #grad_monitors = [
    #    GradNorm(visualizing=True),
    #    ParamNorm( visualizing=True),
    #    UpdateRatio( visualizing=True)
    #]

    monitors = metric_monitors  #+ grad_monitors
    #monitors += viz_monitors

    mc = MonitorCollection('test',*monitors)
    #mc.track_monitor = best_valid_loss
    return mc
