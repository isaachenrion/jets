from src.admin._Administrator import _Administrator

from src.monitors import *
from collections import OrderedDict
class Administrator(_Administrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_monitors(self, epochs, visualizing):
        ''' STATS LOGGER '''
        '''----------------------------------------------------------------------- '''
        #roc_auc = ROCAUC(visualizing=True)
        #inv_fpr = InvFPR(visualizing=True)
        ##best_roc_auc = Best(roc_auc)
        #best_inv_fpr = Best(inv_fpr)
        ##inv_fpr_at_best_roc_auc = LogOnImprovement(inv_fpr, best_roc_auc)
        #roc_auc_at_best_inv_fpr = LogOnImprovement(roc_auc, best_inv_fpr)

        valid_loss = Regurgitate('valid_loss', visualizing=True)
        metric_monitors = [
            #roc_auc,
            #inv_fpr,
            ###best_roc_auc,
            #best_inv_fpr,
            ##inv_fpr_at_best_roc_auc,
            #roc_auc,
            #roc_auc_at_best_inv_fpr,
            #best_roc_auc,
            ProteinMetrics(k=1,visualizing=True),
            ProteinMetrics(k=2,visualizing=True),
            ProteinMetrics(k=5,visualizing=True),
            ProteinMetrics(k=10,visualizing=True),
            #TopLK(1, visualizing=True),
            #TopLK(2, visualizing=True),
            #TopLK(5, visualizing=True),
            #TopLK(10, visualizing=True),
            #Precision(visualizing=True),
            #Recall(visualizing=True),
            valid_loss,
            Regurgitate('train_loss', visualizing=True)

        ]
        self.metric_monitors = metric_monitors

        time_monitors = [
            Regurgitate('epoch', visualizing=False),
            Regurgitate('iteration', visualizing=False),
            Collect('logtime', fn='last', visualizing=False),
            Hours(),
            Collect('time', fn='sum', visualizing=False),
            ETA(self.start_dt, epochs)
        ]

        model_file = os.path.join(self.exp_dir, 'model_state_dict.pt')
        settings_file = os.path.join(self.exp_dir, 'settings.pickle')
        saver = Saver(valid_loss, model_file, settings_file, visualizing=False)

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
            Collect('lr', fn='last', visualizing=True),
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
