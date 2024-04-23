import torch
from architecture import L2Net, SOSNet32x32, HyNet


def build_general_model(args, arch_type, pretrained_dir):
    if arch_type == "l2net":
        model = L2Net(out_dim=args.out_dim, binary=False)
    elif arch_type == "sosnet":
        model = SOSNet32x32(dim_desc=args.out_dim)
    elif arch_type == "hynet":
        model = HyNet(dim_desc=args.out_dim, drop_rate=args.drop_rate)
    else:
        raise ValueError

    if pretrained_dir is not None:
        model.load_state_dict(torch.load(args.pretrained,
                                         map_location="cpu" if args.cpuonly else torch.device(
                                             "cuda:" + str(args.num_device))))

    return model


def build_model(args):
    # set up the models(list)
    model = build_general_model(args, arch_type=args.arch_type, pretrained_dir=args.pretrained)
    return model
