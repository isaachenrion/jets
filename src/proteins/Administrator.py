from src.admin._Administrator import _Administrator

from src.monitors import *
from collections import OrderedDict

class Administrator(_Administrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_monitors(self):
        if self.train:
            return self.setup_training_monitors()
        else:
            return self.setup_testing_monitors()

    def setup_training_monitors(self):

        valid_loss = Regurgitate('valid_loss', visualizing=True)
        best_valid_loss = Best(valid_loss)
        metric_monitors = [
            #ProteinMetrics(k=1,visualizing=True),
            #ProteinMetrics(k=2,visualizing=True),
            #ProteinMetrics(k=5,visualizing=True),
            #ProteinMetrics(k=10,visualizing=True),
            valid_loss,
            best_valid_loss,
            Regurgitate('train_loss', visualizing=True)

        ]
        self.metric_monitors = metric_monitors

        time_monitors = [
            Regurgitate('epoch', visualizing=False),
            Regurgitate('iteration', visualizing=False),
            Collect('logtime', fn='last', visualizing=False),
            Hours(),
            Collect('time', fn='sum', visualizing=False),
            ETA(self.start_dt, self.epochs)
        ]

        model_file = os.path.join(self.exp_dir, 'model_state_dict.pt')
        settings_file = os.path.join(self.exp_dir, 'settings.pickle')
        saver = Saver(best_valid_loss, model_file, settings_file, visualizing=False)

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

    def setup_testing_monitors(self):
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
        self.metric_monitors = monitors
        monitor_dict = OrderedDict()
        for m in monitors:
            monitor_dict[m.name] = m

        return monitor_dict
