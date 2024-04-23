import optimize


def init_scheduler(args, dataset_loader, optimizer):
    # overwrites args.num_epochs is num_steps is specified
    num_steps = args.num_epochs * len(dataset_loader)
    lr_scheduler = optimize.get_lr_scheduler(optimizer, lr_policy=args.lr_policy,
                                             num_steps=num_steps)
    return lr_scheduler
