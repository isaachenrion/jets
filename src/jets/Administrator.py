from src.admin._Administrator import _Administrator

from src.monitors import *
from collections import OrderedDict
from src.admin.MonitorCollection import MonitorCollection
class Administrator(_Administrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_training_only_monitors()


    def setup_monitors(self):
        if self.train:
            return self.setup_training_monitors()
        else:
            return self.setup_testing_monitors()

    def setup_training_only_monitors(self):
        grad_monitors = [
            GradNorm(fn='last',visualizing=True),
            ParamNorm(fn='last', visualizing=True),
            UpdateRatio(fn='last', visualizing=True)
        ]
        monitors = grad_monitors

        monitor_dict = OrderedDict()
        for m in monitors: monitor_dict[m.name] = m

        self.training_only_monitors = MonitorCollection(**monitor_dict)
        self.training_only_monitors.initialize(self.logger.statsdir, self.logger.plotsdir)

    def setup_training_monitors(self):
        roc_auc = ROCAUC(visualizing=True, ndp=5)
        inv_fpr = InvFPR(visualizing=True)
        best_inv_fpr = Best(inv_fpr)
        roc_auc_at_best_inv_fpr = LogOnImprovement(roc_auc, best_inv_fpr)

        metric_monitors = [
            inv_fpr,
            best_inv_fpr,
            roc_auc,
            roc_auc_at_best_inv_fpr,
            Regurgitate('valid_loss', ndp=3,visualizing=True),
            Regurgitate('train_loss', ndp=3,visualizing=True)

        ]
        self.metric_monitors = metric_monitors

        time_monitors = [
            Regurgitate('epoch', visualizing=False, printing=False),
            Regurgitate('iteration', visualizing=False, printing=False),
            Hours(),
            ETA(self.start_dt, self.epochs)
        ]

        model_file = os.path.join(self.exp_dir, 'model_state_dict.pt')
        settings_file = os.path.join(self.exp_dir, 'settings.pickle')
        saver = Saver(best_inv_fpr, model_file, settings_file, visualizing=False, printing=False)

        admin_monitors = [
            saver,
            ]
        self.saver = saver

        #if torch.cuda.is_available():
        #    admin_monitors += [
        #        Collect('gpu_load',fn='last', visualizing=True),
        #        Collect('gpu_util',fn='last', visualizing=True),
        #        ]

        optim_monitors = [
            Collect('lr', fn='last', ndp=8,visualizing=True),
        ]
        grad_monitors = [
            GradNorm(fn='last',visualizing=True),
            #GradVariance(fn='last', visualizing=True),
            ParamNorm(fn='last', visualizing=True),
            #ParamVariance(fn='last', visualizing=True),
            UpdateRatio(fn='last', visualizing=True)
        ]
        self.grad_monitors = grad_monitors

        monitors = metric_monitors + optim_monitors + time_monitors + admin_monitors

        monitor_dict = OrderedDict()
        for m in monitors: monitor_dict[m.name] = m


        return monitor_dict

    def setup_testing_monitors(self):
        roc_auc = ROCAUC(visualizing=True)
        inv_fpr = InvFPR(visualizing=True)
        best_inv_fpr = Best(inv_fpr)

        monitors = [
            inv_fpr,
            best_inv_fpr,
            roc_auc,
            Regurgitate('test_loss', visualizing=True),
            ]
        self.metric_monitors = monitors

        monitor_dict = OrderedDict()
        for m in monitors:
            monitor_dict[m.name] = m

        monitor_dict['model'] = Collect('model', fn='last', visualizing=False, numerical=False)
        return monitor_dict
