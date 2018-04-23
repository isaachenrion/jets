from src.monitors import *
from src.admin.MonitorCollection import MonitorCollection

def test_monitor_collection():
    test_loss = Regurgitate('valid_loss', visualizing=True)
    #best_test_loss = Best(test_loss)

    monitors = [
        ProteinMetrics(k=1,visualizing=True),
        ProteinMetrics(k=2,visualizing=True),
        ProteinMetrics(k=5,visualizing=True),
        ProteinMetrics(k=10,visualizing=True),
        Regurgitate('model', numerical=False, visualizing=False,printing=True),
        test_loss,
        #best_test_loss,
    ]
    mc = MonitorCollection(*monitors)

    return mc
