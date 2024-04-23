import misc
from tlib import TTools
from datasets import BrownDataset, HPatches
from evaluate import DescriptorEvaluator, GenericLearnedDescriptorExtractor
from transform import get_transform_tensor_c2b


def init_evaluator(args, ttools: TTools, tforms, model, device):
    evaluators = {}
    for dset_dot_seq in args.test_data:
        dset_name, split_name = dset_dot_seq.split(".")
        if dset_name == "HP":
            dset = HPatches(root="./data/hpatches", split_name=split_name, transform=tforms, train=False,
                            data_aug=False, download=True)
            ttools.logger.info(f"adding evaluator {dset_dot_seq}")
            extractor_tforms = get_transform_tensor_c2b()
            desc_extractor = GenericLearnedDescriptorExtractor(patch_size=args.patch_size, model=model,
                                                               batch_size=args.bs,
                                                               transform=extractor_tforms, device=device, )
            evaluators[dset_dot_seq] = DescriptorEvaluator(extractor=desc_extractor, datasets=dset,
                                                           dataset_type=dset_name,
                                                           batch_size=args.bs,
                                                           binarize=False, metric=args.metric,
                                                           out_dim=args.out_dim)
        elif dset_name == "brown":
            dset = BrownDataset(root="./data/brown", name=split_name, download=True, train=False,
                                transform=tforms, data_aug=False)
            ttools.logger.info(f"adding evaluator {dset_dot_seq}")
            desc_extractor = GenericLearnedDescriptorExtractor(patch_size=args.patch_size, model=model,
                                                               batch_size=args.bs,
                                                               transform=None, device=device, )
            evaluators[dset_dot_seq] = DescriptorEvaluator(extractor=desc_extractor, datasets=dset,
                                                           dataset_type=dset_name, batch_size=args.bs,
                                                           binarize=False, metric=args.metric,
                                                           out_dim=args.out_dim)
        else:
            raise ValueError("dataset not recognized")

    return evaluators
