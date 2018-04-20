import os
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
            GradNorm(visualizing=True),
            ParamNorm(visualizing=True),
            UpdateRatio(visualizing=True)
        ]
        viz_monitors = [
            BatchMatrixMonitor('yy', n_epochs=self.passed_args['lf'], batch_size=10, visualizing=True),
            BatchMatrixMonitor('half', n_epochs=self.passed_args['lf'], batch_size=10, visualizing=True),
            BatchMatrixMonitor('hard_pred', n_epochs=self.passed_args['lf'], batch_size=10, visualizing=True),
            BatchMatrixMonitor('yy_pred', n_epochs=self.passed_args['lf'], batch_size=10, visualizing=True)
        ]
        monitors = grad_monitors + viz_monitors

        self.training_only_monitors = MonitorCollection(*monitors)
        self.training_only_monitors.initialize(self.logger.statsdir, os.path.join(self.logger.plotsdir, 'train') )

    def setup_training_monitors(self):

        valid_loss = Regurgitate('valid_loss', ndp=3,visualizing=True)
        best_valid_loss = Best(valid_loss, track='min',ndp=3)
        metric_monitors = [
            ProteinMetricCollection(1,2,5,10,ndp=3,visualizing=True),
            valid_loss,
            best_valid_loss,
            Regurgitate('train_loss', ndp=3,visualizing=True)

        ]
        self.metric_monitors = metric_monitors

        time_monitors = [
            Regurgitate('epoch', visualizing=False,printing=False),
            Regurgitate('iteration', visualizing=False, printing=False),
            Hours(),
            Collect('time', fn='sum', visualizing=False, printing=False),
            ETA(self.start_dt, self.epochs)
        ]

        model_file = os.path.join(self.exp_dir, 'model_state_dict.pt')
        settings_file = os.path.join(self.exp_dir, 'settings.pickle')
        saver = Saver(best_valid_loss, model_file, settings_file, visualizing=False, printing=False)

        admin_monitors = [
            saver,
            ]
        self.saver = saver


        optim_monitors = [
            Collect('lr', fn='last', visualizing=True, ndp=8),
        ]


        viz_monitors = [
            BatchMatrixMonitor('yy', n_epochs=self.passed_args['lf'], batch_size=10, visualizing=True),
            BatchMatrixMonitor('half', n_epochs=self.passed_args['lf'], batch_size=10, visualizing=True),
            BatchMatrixMonitor('hard_pred', n_epochs=self.passed_args['lf'], batch_size=10, visualizing=True),
            BatchMatrixMonitor('yy_pred', n_epochs=self.passed_args['lf'], batch_size=10, visualizing=True)
        ]

        monitors = metric_monitors + optim_monitors + time_monitors + admin_monitors
        monitors += viz_monitors

        return monitors

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

        return monitors
