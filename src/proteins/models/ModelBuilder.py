from src.utils._ModelBuilder import _ModelBuilder
from .wangnet import WangNet
from .graphgen import GraphGen


class ModelBuilder(_ModelBuilder):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def model_dict(self):
        return dict(
            #sg=SparseGraphGen,
            w=WangNet,
            g=GraphGen,
        )

    def construct_model_kwargs(self, args):
        #import ipdb; ipdb.set_trace()
        model_kwargs = {
            # model dimensions
            'features': args.features+1,
            'hidden': args.hidden,

            # logging
            'logging_frequency': args.lf,

            # activation
            'act': args.act,

            # classifier on top
            'predict':args.predict,

            # jet transform
            'model':args.model,

            # NMP
            'iters': args.iters,
            'update': args.update,
            'message': args.message,
            'emb_init':args.emb_init,
            'mp_layer':args.mp,
            'symmetric':not args.asym,
            'readout':args.readout,
            'matrix':args.adj[0] if len(args.adj) == 1 else args.adj,
            'matrix_activation':args.m_act,
            'wn': args.wn,
            'no_grad': args.no_grad,
            'tied': args.tied,

            # Stacked NMP
            'scales': args.scales,
            'pooling_layer':args.pool,
            'pool_first':args.pool_first,

            # Physics NMP
            'alpha':args.alpha,
            'R':args.R,
            'trainable_physics':args.trainable_physics,

            # Physics plus learned NMP
            #'physics_component':args.physics_component,
            'learned_tradeoff':not args.equal_weight,

            # Transformer
            'n_heads':args.n_heads,
            'n_layers':args.n_layers,
            'dq':args.dq,
            'dv':args.dv,
            'dropout':args.model_dropout
        }
        return model_kwargs
