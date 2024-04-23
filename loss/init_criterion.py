from loss import (DynamicSoftMarginLoss, SDNetLoss, SAPDSMLoss, SDDSMLoss, SAPSOSNetLoss, SOSNetLoss, SAPLoss)


def init_criterion(args):
    if args.loss_type == "sdnet":
        criterion = SDNetLoss(margin=args.margin, alpha_init=args.alpha_init, alpha_moment=args.alpha_moment,
                              lambda_clean=args.lambda_clean, lambda_adv=args.lambda_adv)
    else:
        raise ValueError(f"{args.loss_type} is an unknown loss type!")

    return criterion
