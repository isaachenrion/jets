def reset_unused_args(args):

    args.sym = not args.asym

    #if args.jet_transform not in ['stack','one', 'plf', 'lf', 'eye', 'phy', 'nmp', 'physta']:
    #    args.iters = None

    #if args.jet_transform not in ['stack', 'physta']:
    #    args.pool = None
    #    args.scales = None
    #    args.pool_first = None

    #if not args.jet_transform == 'tra':
    #    args.n_layers = None
    #    args.dq = None
    #    args.dv = None
    #    args.n_heads = None

    #if args.jet_transform == 'phy':
    #    args.mp = None
    #    args.matrix = None
    #    args.sym = None
    #    if args.trainable_physics:
    #        #args.alpha = None
    #        args.R = None
    #    #args.alpha = None
    args.learned_tradeoff = not args.equal_weight

    args.experiment_time *= (60 * 60)

    args.train = True





    #if args.pileup:
    #    args.dataset = 'pileup'
    #else:
    #    args.dataset = 'original'

    return args
