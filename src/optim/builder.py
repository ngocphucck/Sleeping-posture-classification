from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import LambdaLR, StepLR, ExponentialLR


def create_optimizer(name, parameters, **kwargs):
    assert name in ['sgd', 'adam', 'rmsprop']

    if name == 'sgd':
        return SGD(parameters, **kwargs)
    elif name == 'adam':
        return Adam(parameters, **kwargs)
    elif name == 'rmsprop':
        return RMSprop(parameters, **kwargs)


def create_lr_scheduler(name, optimizer, **kwargs):
    assert name in ['lambda_lr', 'step_lr', 'exponential_lr']

    if name == 'lambda_lr':
        return LambdaLR(optimizer, **kwargs)
    elif name == 'step_lr':
        return StepLR(optimizer, **kwargs)
    elif name == 'exponential_lr':
        return ExponentialLR(optimizer, **kwargs)
