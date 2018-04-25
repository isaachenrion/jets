from src.monitors import *
from src.admin.MonitorCollection import MonitorCollection

def get_monitor_collections():
    return dict(test=test_monitor_collection())

def test_monitor_collection():
    roc_auc = ROCAUC(visualizing=True)
    inv_fpr = InvFPR(visualizing=True)
    best_inv_fpr = Best(inv_fpr)

    monitors = [
        inv_fpr,
        best_inv_fpr,
        roc_auc,
        Collect('valid_loss', visualizing=True),
        Collect('model', visualizing=False, numerical=False)
        ]
    mc = MonitorCollection(*monitors)

    return mc
