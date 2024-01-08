# Copyright 2021 Zhongyang Zhang
# Contact: mirakuruyoo@gmai.com
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

""" This main entrance of the whole project.

    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and 
    DInterface can be seen as transparent to all your args.    
"""
import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger,CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from model import MInterface
from data import DInterface
from utils.utils import load_model_path_by_args
from utils.misc import load_config
from utils.callbacks import CodeSnapshotCallback, ConfigSnapshotCallback, CustomProgressBar
from datetime import datetime

def load_callbacks(config):
    callbacks = []
    callbacks += [
            ModelCheckpoint(
                dirpath=config.ckpt_dir,
                **config.checkpoint
            ),
            LearningRateMonitor(logging_interval='step'),
            CodeSnapshotCallback(
                config.code_dir, use_version=False
            ),
            ConfigSnapshotCallback(
                config, config.config_dir, use_version=False
            ),
            CustomProgressBar(refresh_rate=1),
        ]
    
    if config.model.scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks


def main(config):
    pl.seed_everything(config.seed)
    load_path = load_model_path_by_args(config.cmd_args)

    data_module = DInterface(**config.dataset)
    model = MInterface(**config.model)


    callbacks = load_callbacks(config)
    loggers = []
        
    if load_path is None:
        loggers += [
                #TensorBoardLogger(args.log_dir, name=args.model_name, version=args.trial_name),
                WandbLogger(save_dir=config.exp_dir, name=config.trial_name, project = config.name)
                #CSVLogger(save_dir=config.cmd_args.log_dir, name=config.name, version=config.cmd_args.trial_name)
            ]
    else:
        loggers += [
                #TensorBoardLogger(args.log_dir, name=args.model_name, version=args.trial_name),
                WandbLogger(save_dir=config.exp_dir, name=config.trial_name, project = config.name, version = config.cmd_args.load_ver, resume = True)
                #CSVLogger(save_dir=config.cmd_args.log_dir, name=config.name, version=config.cmd_args.trial_name)
            ]
    trainer = Trainer(**config.trainer,
                      logger=loggers,
                      callbacks = callbacks)
    if load_path is None:
        trainer.fit(model, data_module)
    else:
        trainer.fit(model, data_module, ckpt_path=load_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control


   
    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Training Info
    parser.add_argument('--log_dir', default='lightning_logs', type=str)
    parser.add_argument('--trial_name', default='test', type=str)

    # Other
    parser.add_argument('--config', required=True, help='path to config file')
    parser.add_argument('--exp_dir', default='./exp')
    # Add pytorch lightning's args to parser as a group.
    #parser = Trainer.add_argparse_args(parser)

    ## Deprecated, old version
    # parser = Trainer.add_argparse_args(
    #     parser.add_argument_group(title="pl.Trainer args"))

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=100)
    

    args, extras = parser.parse_known_args()



    config = load_config(args.config, cli_args=extras)
    config.cmd_args = vars(args)


    if config.cmd_args.load_ver is None:
        config.trial_name = config.get('trial_name') or (config.tag + datetime.now().strftime('@%Y%m%d-%H%M%S'))
    else:
        config.trial_name =  config.cmd_args.load_ver
     
    config.exp_dir = config.get('exp_dir') or os.path.join(args.exp_dir, config.name)
    config.model.save_dir = config.get('save_dir') or os.path.join(config.exp_dir, config.trial_name, 'save')
    config.ckpt_dir = config.get('ckpt_dir') or os.path.join(config.exp_dir, config.trial_name, 'ckpt')
    config.code_dir = config.get('code_dir') or os.path.join(config.exp_dir, config.trial_name, 'code')
    config.config_dir = config.get('config_dir') or os.path.join(config.exp_dir, config.trial_name, 'config')
    
    main(config)
