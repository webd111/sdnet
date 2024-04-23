from datasets import BrownDataset, HPatches
import transform


def init_data_transform(args):
    dset_means = []
    dset_stds = []
    dset_lengths = []

    for dset_str in args.train_data:
        parts = dset_str.split(".")
        dset_name, seq_name = dset_str.split(".")
        if dset_name == "brown":
            dset = BrownDataset
        elif dset_name == "HP":
            dset = HPatches
        else:
            raise ValueError(f"{dset_name} is not a test dataset")

    if not args.arch_type == "hynet":
        dset_means.append(dset.mean[seq_name])
        dset_stds.append(dset.std[seq_name])
        dset_lengths.append(dset.length[seq_name])

    if args.arch_type == "hynet":
        tforms = transform.get_input_transform_no_norm(args.patch_size)
    else:
        avg_mean, avg_std = transform.compute_multi_dataset_mean_std(dset_lengths, dset_means, dset_stds)
        tforms = transform.get_basic_input_transform(args.patch_size, avg_mean, avg_std)

    return tforms
