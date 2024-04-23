import sys
import torch
import numpy as np

import misc  # 杂项
from datasets import BrownDataset, HPatches


def init_dataset(args, tforms):
    assert args.train_data, misc.yellow("training data is empty")

    train_dsets = []
    for dset_str in args.train_data:
        dset_name, split_name = dset_str.split(".")
        if dset_name == "HP":
            dset = HPatches(root="./data/hpatches", num_types=16, transform=tforms,
                            data_aug=not args.no_data_aug, download=True, split_name=split_name)
        elif dset_name == "brown":
            dset = BrownDataset(root="./data/brown", name=split_name, download=True,
                                train=True, transform=tforms, triplet=False,
                                data_aug=not args.no_data_aug)
        else:
            raise ValueError("dataset not recognized")
        train_dsets.append(dset)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(train_dsets), batch_size=args.bs,
                                               shuffle=not args.no_shuffle,
                                               num_workers=0 if sys.platform == "win32" else 8,
                                               drop_last=True, pin_memory=True,
                                               worker_init_fn=lambda x: np.random.seed(np.random.get_state()[1][0] + x))

    # overwrites args.test_freq & args.test_epoch_freq if args.test_every_epoch is set
    if hasattr(args, 'test_epoch_freq'):
        if args.test_epoch_freq is not None:
            args.test_freq = len(train_loader) * args.test_epoch_freq
    if args.test_every_epoch:
        args.test_freq = len(train_loader)
    # overwrites args.save_freq if args.save_every_epoch is set
    if hasattr(args, 'save_epoch_freq'):
        if args.save_epoch_freq is not None:
            args.save_freq = len(train_loader) * args.save_epoch_freq
    if args.save_every_epoch:
        args.save_freq = len(train_loader)
    if hasattr(args, 'test_HP_epoch_freq'):
        if args.test_HP_epoch_freq is not None:
            args.test_HP_freq = len(train_loader) * args.test_HP_epoch_freq

    return train_loader
