def reset_unused_args(args):
    if args.jet_transform not in ['stack','one', 'eye', 'phy', 'nmp', 'physta']:
        args.iters = None

    if args.jet_transform not in ['stack', 'physta']:
        args.pool = None
        args.scales = None
        args.pool_first = None

    if not args.jet_transform == 'tra':
        args.n_layers = None
        args.dq = None
        args.dv = None
        args.n_heads = None

    if args.jet_transform == 'phy':
        args.mp = None
        args.matrix = None
        args.sym = None
        if args.trainable_physics:
            #args.alpha = None
            args.R = None
        #args.alpha = None

    args.train = True
    if args.debug:
        args.no_email = True
        args.hidden = 7
        args.batch_size = 5
        args.verbose = True
        args.epochs = 3
        args.n_train = 1000
        args.seed = 1


    if args.n_train <= 5 * args.n_valid and args.n_train > 0:
        args.n_valid = args.n_train // 5

    if args.pileup:
        args.dataset = 'pileup'
    else:
        args.dataset = 'original'
    return args
