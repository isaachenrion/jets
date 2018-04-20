import torch
from .meta import Collect

class GradNorm(Collect):
    def __init__(self, **kwargs):
        super().__init__('grad_norm', fn='last', **kwargs)

    def call(self, grads=None, **kwargs):
        #grads = [p.grad for p in model.parameters()]
        #grads = torch.cat([g.view(-1) for g in grads],0)

        grad_norm = torch.norm(grads)
        super().call(grad_norm=grad_norm)
        return self.value

class ParamNorm(Collect):
    def __init__(self, **kwargs):
        super().__init__('param_norm', fn='last', **kwargs)

    def call(self, model_params=None, **kwargs):
        #model_params = list(model.parameters())
        #model_params = torch.cat([p.view(-1) for p in model_params],0)

        param_norm = torch.norm(model_params)
        super().call(param_norm=param_norm)
        return self.value

class GradVariance(Collect):
    def __init__(self, **kwargs):
        super().__init__('grad_variance', fn='last', **kwargs)

    def call(self, grads=None, **kwargs):
        #grads = [p.grad for p in model.parameters()]
        #grads = torch.cat([g.view(-1) for g in grads],0)
        grad_variance = torch.var(grads)
        super().call(grad_variance=grad_variance)
        return self.value


class ParamVariance(Collect):
    def __init__(self, **kwargs):
        super().__init__('param_variance', fn='last', **kwargs)

    def call(self, model_params=None, **kwargs):
        #model_params = list(model.parameters())
        #model_params = torch.cat([p.view(-1) for p in model_params],0)
        param_variance = torch.var(model_params)
        super().call(param_variance=param_variance)
        return self.value

class UpdateRatio(Collect):
    def __init__(self, **kwargs):
        super().__init__('update_ratio', fn='last', **kwargs)

    def call(self, model_params=None, old_params=None, **kwargs):
        #model_params = list(model.parameters())
        #model_params = torch.cat([p.view(-1) for p in model_params],0)
        #param_variance = torch.var(model_params)
        updates = model_params - old_params
        update_norm = torch.norm(updates)
        param_norm = torch.norm(model_params)
        update_ratio = update_norm / param_norm
        super().call(update_ratio=update_ratio)
        return self.value
