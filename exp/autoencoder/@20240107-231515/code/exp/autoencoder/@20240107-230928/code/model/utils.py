
import torch
import torch.nn as nn
from pytorch_lightning.utilities.rank_zero import rank_zero_debug

def getattr_recursive(m, attr):
    for name in attr.split('.'):
        m = getattr(m, name)
    return m

def get_parameters(model, name):
    module = getattr_recursive(model, name)
    if isinstance(module, nn.Module):
        return module.parameters()
    elif isinstance(module, nn.Parameter):
        return module
    return []


def parse_optimizer(config, model):
    if hasattr(config, 'params'):
        params = [{'params': get_parameters(model, name), 'name': name, **args} for name, args in config.params.items()]
        rank_zero_debug('Specify optimizer params:', config.params)
    else:
        params = model.parameters()
    if config.name in ['FusedAdam']:
        import apex
        optim = getattr(apex.optimizers, config.name)(params, **config.args)
    else:
        optim = getattr(torch.optim, config.name)(params, **config.args)
    return optim
