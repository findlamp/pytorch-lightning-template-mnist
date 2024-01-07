# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities.rank_zero import rank_zero_debug


class MInterface(pl.LightningModule):
    def __init__(self, name, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        out = self(x)
        loss = self.loss_function(out, x)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        out = self(x)
        loss = self.loss_function(out, x)
       
       
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        #optimizer = torch.optim.Adam(
        #    self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)
        print(self.hparams.optimizer)
        optimizer = parse_optimizer(self.hparams.optimizer, self.model)
        if self.hparams.scheduler is None:
            return optimizer
        else:
            if self.hparams.scheduler.name == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.scheduler.lr_decay_steps,
                                       gamma=self.hparams.scheduler.lr_decay_rate)
            elif self.hparams.scheduler.name == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.scheduler.lr_decay_steps,
                                                  eta_min=self.hparams.scheduler.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        elif loss == 'bce':
            self.loss_function = F.binary_cross_entropy
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        name = self.hparams.name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)

    
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


