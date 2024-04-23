from tlib.tools import TTools
import os.path
import time
import numpy as np
import torch
import torch.nn as nn
import wandb
from architecture import build_model
from loss import init_criterion
from transform import init_data_transform
from datasets import init_dataset
from evaluate import init_evaluator
from optimize import init_scheduler
from generator import Generator
import misc
import optimize
from torch.cuda.amp import autocast, GradScaler

os.environ["WANDB_API_KEY"] = "60b9c7d913c7d05df8e5a73f0a656ad391949a88"


def config(ttools: TTools, ):
    # standard training params
    ttools.add_arg("--cpuonly", action="store_true")
    ttools.add_arg("--debug", action="store_true")
    ttools.add_arg("--half", action="store_true")
    # ttools.add_arg("--num_device", type=int, default=0)
    ttools.add_arg("--num_epochs", type=int, default=10)
    ttools.add_arg("--num_steps", type=int, default=0)
    # ttools.add_arg("--test_freq", type=int, default=5000)
    ttools.add_arg("--test_every_epoch", action="store_true")
    # ttools.add_arg("--test_epoch_freq", type=int, default=None)
    # ttools.add_arg("--test_HP_freq", type=int, default=5000)
    ttools.add_arg("--test_HP_epoch_freq", type=int, default=10)
    ttools.add_arg("--print_freq", type=int, default=10)
    ttools.add_arg("--save_freq", type=int, default=5000)
    ttools.add_arg("--save_every_epoch", action="store_true")
    ttools.add_arg("--save_epoch_freq", type=int, default=None)
    # ttools.add_arg("--save_every_model", action="store_true")
    # optimizer params
    ttools.add_arg("--bs", type=int, default=1024)  # ESSENTIAL
    ttools.add_arg("--optim", type=str, default="sgd")  # ESSENTIAL
    ttools.add_arg("--lr", type=float, default=0.1)  # ESSENTIAL
    ttools.add_arg("--lr_policy", type=str, default="cos")  # ESSENTIAL
    ttools.add_arg("--momentum", type=float, default=0.9)
    ttools.add_arg("--dampening", type=float, default=0)
    ttools.add_arg("--wd", type=float, default=0.0001)
    ttools.add_arg("--output_root", type=str, default="./trained")
    # pre-processing transform
    ttools.add_arg("--log_name", type=str, default="default")
    ttools.add_arg("--suffix", type=str, default="")
    # architecture params
    ttools.add_arg("--arch_type", type=str, default="hynet")  # ESSENTIAL
    ttools.add_arg("--pretrained", type=str, default=None)
    ttools.add_arg("--out_dim", type=int, default=128)
    ttools.add_arg("--drop_rate", type=float, default=0.3)
    # loss params
    ttools.add_arg("--loss_type", type=str, default="dsm")  # ESSENTIAL
    ttools.add_arg("--margin", type=float, default=0.45)
    ttools.add_arg("--alpha_init", type=float, default=0)
    ttools.add_arg("--alpha_moment", type=float, default=0.999)
    ttools.add_arg("--lambda_clean", type=float, default=1.0)
    ttools.add_arg("--lambda_adv", type=float, default=1.0)
    ## for adv training
    ttools.add_arg("--adv_step", type=float, default=0.03)  # ESSENTIAL
    ttools.add_arg("--adv_iter", type=int, default=3)  # ESSENTIAL
    # data params
    # Add "HP.full" if full hpatches dataset is desired
    ttools.add_arg("--train_data", nargs="+", type=str, default=["brown.liberty"])
    ttools.add_arg("--test_data", nargs="+", type=str, default=["brown.yosemite"])
    ttools.add_arg("--patch_size", type=int, default=32)
    ttools.add_arg("--metric", type=str, default="l2")  # ESSENTIAL
    ttools.add_arg("--no_shuffle", action="store_true")
    ttools.add_arg("--no_data_aug", action="store_true")
    # ttools.add_arg("--output_index", action="store_true")

    args = ttools.get_args()
    if not os.path.isdir('log'):
        os.mkdir(path='log')
    if not os.path.isdir('trained'):
        os.mkdir(path='trained')

    hparams = dict(
        # basic
        architecture=args.arch_type,
        out_dim=args.out_dim,
        loss=args.loss_type,
        margin=args.margin,
        metric=args.metric,
        no_data_aug=args.no_data_aug,
        train_data=args.train_data,
        test_data=args.test_data,
        batchsize=args.bs,
        optim=args.optim,
        lr=args.lr,
        lr_policy=args.lr_policy,
        drop_rate=args.drop_rate,
        # adv training
        lambda_clean=args.lambda_clean,
        lambda_adv=args.lambda_adv,
        adv_iter=args.adv_iter,
        adv_step=args.adv_step,
    )

    name = args.arch_type + "_bs" + str(args.bs) + "_ep" + str(args.num_epochs) \
           + "_loss_" + args.loss_type + "_odim" + str(args.out_dim)

    return args, hparams, name


def init_file(args, ttools: TTools, ):
    # make output folder
    s = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
    s += "_" + "TRAIN"
    if len(args.train_data) < 3:
        for t in args.train_data:
            s += "_" + t
    else:
        s += "_" + f"{len(args.train_data)}dsets"
    s += "_" + args.arch_type
    s += "_" + args.loss_type
    s += "_" + args.suffix if args.suffix != "" else ""
    args.save_dir = f"{args.output_root}/{s}"

    ttools.logger.info(misc.green("result path = " + args.save_dir))
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    # save arguments
    with open(os.path.join(args.save_dir, "args.txt"), "w") as f:
        for args_name, args_value in vars(args).items():
            f.write('%s=%s\n' % (args_name, args_value))
        f.close()

    # # save mean std
    # torch.save({"mean": avg_mean, "std": avg_std},
    #            os.path.join(args.save_dir, "mean_std.pt"))


def test(ttools: TTools, evaluators, dataset_type):
    ttools.logger.info("running evaluation...")
    fpr95 = {}
    mAP = {}
    APHP = {}
    mAPHP = {}
    for dset_dot_seq, evaluator in evaluators.items():
        if dset_dot_seq.split(".")[0] != dataset_type:
            continue
        evaluator.run()
        if dset_dot_seq.split(".")[0] == "brown":
            fpr95[dset_dot_seq] = evaluator.computeFPR95()
            mAP[dset_dot_seq] = evaluator.computeMatchingScore()
        elif dset_dot_seq.split(".")[0] == "HP":
            # APHP[dset_dot_seq] = evaluator.computeVerificationScore()
            mAPHP[dset_dot_seq] = evaluator.computeMatchingScore()
    return fpr95, mAP, APHP, mAPHP


def train(args, ttools: TTools, train_loader, model, device, optimizer, criterion, lr_scheduler, evaluators):
    x = 0
    epochs = args.num_epochs
    steps = args.num_epochs * len(train_loader)
    best_result = 1
    last_best_model_dir = None
    advg = Generator()
    scaler = GradScaler()

    for epoch_idx in range(epochs):
        np.random.seed()
        for batch_idx, batch_data in enumerate(train_loader):
            step = epoch_idx * len(train_loader) + batch_idx
            if step >= steps:  # stop
                break

            # if use adversarial generation
            adv_dict = advg.adv_gen(x=batch_data, adv_model=model,
                                    iters=args.adv_iter, device=device, step=args.adv_step,
                                    output_rec=True, output_confidence=True, half=args.half)
            data = torch.cat(batch_data, dim=0).to(device)  # data for net
            for key in adv_dict.keys():
                if key.startswith("x_") and adv_dict[key] is not None:  # if value is data
                    data = torch.cat([data, adv_dict[key]], dim=0)

            # forward-loss-backprop
            optimizer.zero_grad()
            if args.half:
                with autocast():
                    if args.arch_type == "hynet":
                        out = model(data, is_training=True)
                    else:
                        out = model(data)
            else:
                if args.arch_type == "hynet":
                    out = model(data, is_training=True)
                else:
                    out = model(data)

            dict_loss_in = {"descs": out}
            dict_loss_in.update({"current_epoch": epoch_idx})
            if adv_dict is not None:
                dict_loss_in.update(adv_dict)

            dict_loss_out = criterion(dict_loss_in)
            loss = dict_loss_out["loss"]

            # check whether loss is NaN
            if torch.isnan(loss):
                if isinstance(model, nn.DataParallel):
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                torch.save(state_dict, f"{args.save_dir}/model_loss_is_nan{epoch_idx}.state_dict")
                torch.save(state_dict, os.path.join(wandb.run.dir, 'model.state_dict'))
                raise ArithmeticError

            if args.half:
                with autocast():
                    scaler.scale(loss).backward()
                    # clip model parameters
                    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2.0)
                    scaler.step(optimizer)
                    scaler.update()
            else:
                loss.backward()
                # clip model parameters
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2.0)
                optimizer.step()

            # check whether model parameters have NaN
            for param in model.parameters():
                if torch.sum(torch.isnan(param)):
                    raise ArithmeticError

            # change learning rate
            if lr_scheduler is not None:
                lr_scheduler.step()

            # print result
            if (step + 1) % args.print_freq == 0:
                if lr_scheduler is not None:
                    lr_current = lr_scheduler.get_last_lr()[0]
                else:
                    lr_current = args.lr
                ttools.logger.warning(f"Epoch {epoch_idx} ({batch_idx}/{len(train_loader)}) | "
                                      f"lr={lr_current} | Loss={loss.item():.3f}")
                x += 1
                wandb_log_dict = {"Epoch": epoch_idx, "LR": lr_current}
                wandb_log_dict.update(dict_loss_out)
                wandb.log(wandb_log_dict)

            # brown dataset test
            if (step + 1) % args.test_freq == 0:
                model.eval()
                test_result = test(ttools, evaluators, dataset_type="brown")
                fpr95s = []
                mAPs = []
                # brown dataset
                for dset_dot_seq, _ in test_result[0].items():
                    fpr95 = test_result[0][dset_dot_seq]
                    mAP = test_result[1][dset_dot_seq]
                    fpr95s.append(fpr95)
                    mAPs.append(mAP)
                    ttools.logger.warning(f"FPR95-{dset_dot_seq} = {fpr95 * 100}%, mAP-{dset_dot_seq} = {mAP * 100}%")

                model.train()
                wandb.log({"Test": np.mean(np.array(fpr95s)), "Test mAP": np.mean(np.array(mAPs)), })

                # Using brown Verification result as best result criterion
                fpr95_mean = np.mean(np.array(fpr95s))
                if best_result > fpr95_mean:
                    best_result = fpr95_mean
                    best_epoch = epoch_idx
                    wandb.log({f"Best Result Brown": best_result})
                    if isinstance(model, nn.DataParallel):
                        state_dict = model.module.state_dict()
                    else:
                        state_dict = model.state_dict()
                    if last_best_model_dir is not None:
                        os.remove(last_best_model_dir)
                    last_best_model_dir = os.path.join(wandb.run.dir,
                                                       f'best_model_ep{best_epoch}_fpr{best_result:.3f}.state_dict')
                    torch.save(state_dict, last_best_model_dir)

            # HP dataset test
            if (step + 1) % args.test_HP_freq == 0:
                model.eval()
                test_result = test(ttools, evaluators, dataset_type="HP")
                mAPHPs = []
                # hpatches dataset
                for dset_dot_seq, _ in test_result[3].items():
                    mAPHP = test_result[3][dset_dot_seq]
                    mAPHPs.append(mAPHP)
                    ttools.logger.warning(f"mAP-{dset_dot_seq} = {mAPHP * 100}%")

                model.train()
                wandb.log({"Test mAP HP": np.mean(np.array(mAPHPs)), })

            # save model
            if (step + 1) % args.save_freq == 0:
                if not torch.isnan(loss).item():
                    if isinstance(model, nn.DataParallel):
                        state_dict = model.module.state_dict()
                    else:
                        state_dict = model.state_dict()
                    torch.save(state_dict, f"{args.save_dir}/model.state_dict")
                    torch.save(state_dict, os.path.join(wandb.run.dir, 'model.state_dict'))


if __name__ == '__main__':
    ttools = TTools()
    args, hparams, name = config(ttools=ttools)

    # select device
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    device = "cpu" if args.cpuonly else torch.device("cuda")
    if device != "cpu":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # set up the models
    model = build_model(args)
    model.to(device)
    if not args.cpuonly and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # wandb
    wandb.init(project="sapnet_ablation", config=hparams, name=name, sync_tensorboard=True)
    wandb.watch(model)

    optimizer = optimize.create_optimizer(optimizer_type=args.optim, model_params=model.parameters(),
                                          lr=args.lr, momentum=args.momentum)

    # set up criterion
    criterion = init_criterion(args).to(device)

    # set up training and validation data
    tforms = init_data_transform(args)
    train_loader = init_dataset(args, tforms)

    # set up evaluators
    evaluators = init_evaluator(args, ttools, tforms=tforms, model=model, device=device)

    # set up scheduler
    lr_scheduler = init_scheduler(args, dataset_loader=train_loader, optimizer=optimizer)

    # set up file system
    init_file(args, ttools)

    # begin training
    start_train_t = time.time()
    train(args, ttools, train_loader, model, device, optimizer, criterion, lr_scheduler, evaluators)
    hours_took = (time.time() - start_train_t) / 3600.0
    ttools.logger.warning(f"training took {hours_took:.2f} hours")

    # Save final model
    if isinstance(model, nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, f"{args.save_dir}/model.state_dict")
    torch.save(state_dict, os.path.join(wandb.run.dir, 'model.state_dict'))
    # save arguments for wandb
    with open(os.path.join(wandb.run.dir, "args.txt"), "w") as f:
        for args_name, args_value in vars(args).items():
            f.write('%s=%s\n' % (args_name, args_value))
        f.close()

    hours_took = (time.time() - start_train_t) / 3600.0
    ttools.logger.warning(f"training took {hours_took:.2f} hours")
