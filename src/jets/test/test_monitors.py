from src.monitors import *
from src.admin.MonitorCollection import MonitorCollection

def get_monitor_collections():
    return dict(test=test_monitor_collection())

def test_monitor_collection():

    roc_auc = ROCAUC(visualizing=False, ndp=5)
    mean_roc_auc = Mean(roc_auc, ndp=5)
    std_roc_auc = Std(roc_auc, ndp=5)

    inv_fpr = InvFPR(visualizing=False)
    mean_inv_fpr = Mean(inv_fpr)
    std_inv_fpr = Std(inv_fpr)

    valid_loss = Collect('loss', ndp=3,visualizing=False)
    mean_valid_loss = Mean(valid_loss, ndp=3)
    std_valid_loss = Std(valid_loss, ndp=3)

    metric_monitors = [
        inv_fpr,
        mean_inv_fpr,
        std_inv_fpr,

        roc_auc,
        mean_roc_auc,
        std_roc_auc,

        valid_loss,
        mean_valid_loss,
        std_valid_loss,

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
