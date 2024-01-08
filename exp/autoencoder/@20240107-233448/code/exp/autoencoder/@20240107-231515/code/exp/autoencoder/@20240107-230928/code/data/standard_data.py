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

import torch
import os.path as op
import numpy as np
import pickle as pkl
import torch.utils.data as data

from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split


class StandardData(data.Dataset):
    def __init__(self, root_dir=r'MNIST',
                 class_num=9,
                 train=True,
                 no_augment=True,
                 aug_prob=0.5,
                 img_mean=(0.485, 0.456, 0.406),
                 img_std=(0.229, 0.224, 0.225)):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.aug = train and not no_augment

        self.dataset = datasets.MNIST(root=root_dir, download=True, train=train, transform= transforms.ToTensor())
        

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
       
        
        return self.dataset[idx]