# From original author
import torch.optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR, _LRScheduler, CosineAnnealingLR


# Linearly decrease the learning rate to zero
class DecreaseToZero(_LRScheduler):
    def __init__(self, optimizer, max_steps, last_epoch=-1):
        self.max_steps = max_steps
        super(DecreaseToZero, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        eps = 1e-6
        return [
            max(base_lr * (1.0 - float(self.last_epoch) / float(self.max_steps)), eps)
            for base_lr in self.base_lrs
        ]


def get_lr_scheduler(optimizer: object, lr_policy: object, num_steps: object = None, num_epoch: object = None) -> object:
    if lr_policy is None:
        return None

    parts = lr_policy.split("_")
    policy = parts[0]
    if len(parts) > 1:
        params = parts[1:]

    if policy == "step":
        assert len(params) == 2, "StepLR() needs 2 parameters"
        lr_scheduler = StepLR(optimizer, step_size=int(params[0]), gamma=float(params[1]))
    elif policy == "exp":
        assert len(params) == 1, "ExponentialLR() needs decay rate"
        lr_scheduler = ExponentialLR(optimizer, gamma=float(params[0]))
    elif policy == "linear":
        assert num_steps is not None, "DecreaseToZero() needs number of steps"
        lr_scheduler = DecreaseToZero(optimizer, max_steps=num_steps)
    elif policy == "cos":
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_steps)
    else:
        lr_scheduler = None
    return lr_scheduler


def create_optimizer(optimizer_type, model_params, lr, wd=0.0001, momentum=0.9, dampening=0):
    # assert len(model_params) > 0, 'model does not contain parameters!'
    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model_params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=wd)
    elif optimizer_type == "adadelta":
        optimizer = torch.optim.Adadelta(model_params, lr=lr, rho=0.9, eps=1e-06, weight_decay=wd)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(model_params, lr=lr, betas=(0.5, 0.9), eps=1e-08, weight_decay=wd)
    else:
        raise ValueError("double check optimizer!")

    return optimizer