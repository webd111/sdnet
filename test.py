import argparse
import logging
import os.path
import torch
from datasets import BrownDataset, HPatches
from evaluate import DescriptorEvaluator, GenericLearnedDescriptorExtractor
from architecture import L2Net, SOSNet32x32, HyNet
import transform


logging.basicConfig(
    format="%(asctime)s %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

def add_arg(*args, **kwargs):
    kwargs["help"] = "(default: %(default)s)"
    if not kwargs.get("type", bool) == bool:
        kwargs["metavar"] = ""
    parser.add_argument(*args, **kwargs)


add_arg("--cpuonly", action="store_true")
add_arg("--bs", type=int, default=1024)
add_arg("--out_dim", type=int, default=128)
add_arg("--model_dir", type=str, default=None)
add_arg("--binary", action="store_true")
add_arg("--test_data", nargs="+", type=str, default=["brown.liberty"])
add_arg("--patch_size", type=int, default=32)
add_arg("--metric", type=str, default="l2")
add_arg("--arch_type", type=str, default="l2net")

args = parser.parse_args()

# select device
device = "cpu" if args.cpuonly else "cuda"

if args.arch_type == "l2net" or args.arch_type == "hardnet":
    model = L2Net(out_dim=256 if args.binary else 128, binary=args.binary)
elif args.arch_type == "sosnet":
    model = SOSNet32x32()
elif args.arch_type == "hynet":
    model = HyNet(dim_desc=args.out_dim)
else:
    raise NotImplementedError

assert args.model_dir is not None, "model directory not specified"

if args.arch_type == "hynet" or args.arch_type == "l2net" or args.arch_type == "sosnet":
    model.load_state_dict(torch.load(os.path.join(args.model_dir), map_location='cuda:0'))
elif args.arch_type == "hardnet":
    model.load_state_dict(torch.load(os.path.join(args.model_dir), map_location='cuda:0')["state_dict"])
else:
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "model.state_dict"), map_location='cuda:0'))

model = model.to(device)
model.eval()

mean_std = torch.load(os.path.join("./pretrained/liberty_float", "mean_std.pt"))

if args.arch_type == "hynet":
    tforms = transform.get_input_transform_no_norm(args.patch_size)
else:
    tforms = transform.get_basic_input_transform(args.patch_size, mean_std["mean"], mean_std["std"])
# tforms = transform.get_basic_input_transform(args.patch_size, mean_std["mean"], mean_std["std"])

evaluators = {}
for dset_dot_seq in args.test_data:
    dset_name, seq_name = dset_dot_seq.split(".")
    if dset_name == "brown":
        dset = BrownDataset(root="./data/brown", name=seq_name, download=True, train=False,
                            transform=tforms, data_aug=False)
        logger.info(f"adding evaluator {dset_dot_seq}")
        desc_extractor = GenericLearnedDescriptorExtractor(patch_size=args.patch_size, model=model, batch_size=args.bs,
                                                           transform=None, device=device)
        evaluators[dset_dot_seq] = DescriptorEvaluator(extractor=desc_extractor, datasets=dset, metric=args.metric,
                                                       batch_size=args.bs, binarize=args.binary)
    elif dset_name == "HP":
        dset = HPatches(root="./data/hpatches", split_name=seq_name, transform=tforms, train=False,
                        data_aug=False, download=True)
        logger.info(f"adding evaluator {dset_dot_seq}")
        extractor_tforms = transform.get_transform_tensor_c2b()
        desc_extractor = GenericLearnedDescriptorExtractor(patch_size=args.patch_size, model=model,
                                                           batch_size=args.bs,
                                                           transform=extractor_tforms, device=device, )
        evaluators[dset_dot_seq] = DescriptorEvaluator(extractor=desc_extractor, datasets=dset, metric=args.metric,
                                                       dataset_type=dset_name,
                                                       batch_size=args.bs,
                                                       binarize=args.binary)
    else:
        raise ValueError("dataset not recognized")


def test():
    logger.info("running evaluation...")
    fpr95 = {}
    AP = {}
    mAP = {}
    for dset_dot_seq, evaluator in evaluators.items():
        evaluator.run()
        fpr95[dset_dot_seq] = evaluator.computeFPR95()
        mAP[dset_dot_seq] = evaluator.computeMatchingScore()
    return fpr95, AP, mAP


test_result = test()
for dset_dot_seq, fpr95 in test_result[0].items():
    if fpr95 is not None:
        logger.info(f"FPR95-{dset_dot_seq} = {fpr95 * 100}%")
    else:
        logger.info(f"FPR95-{dset_dot_seq} is None")
for dset_dot_seq, AP in test_result[1].items():
    if AP is not None:
        logger.info(f"AP-{dset_dot_seq} = {AP * 100}%")
    else:
        logger.info(f"AP-{dset_dot_seq} is None")
for dset_dot_seq, mAP in test_result[2].items():
    logger.info(f"mAP-{dset_dot_seq} = {mAP * 100}%")
