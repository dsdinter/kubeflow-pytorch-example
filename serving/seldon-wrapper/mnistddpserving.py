#!/usr/bin/env python
'''
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import os
import sys
import torch
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.nn.modules import Module

from math import ceil
from random import Random
from torch.autograd import Variable
from torchvision import datasets, transforms

class Net(nn.Module):
    """ Network architecture. """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class mnistddpserving(object):
    def __init__(self):
        self.class_names = ["class:{}".format(str(i)) for i in range(10)]
        self.model = Net()
        # TODO parametrise path to load model
        self.model.load_state_dict(torch.load("/mnt/kubeflow-gcfs/pytorch/model/model.dat",
                                              map_location='cpu'))
        # Ensure the model is in eval/inference mode
        self.model.eval()

    def predict(self, X, feature_names):
        tensor = torch.from_numpy(X).view(-1, 28, 28)
        t = transforms.Normalize((0.1307,), (0.3081,))
        tensor_norm = t(tensor)
        tensor_norm = tensor_norm.unsqueeze(0)
        out = self.model(tensor_norm.float())
        predictions = torch.nn.functional.softmax(out)
        print(predictions)
        return predictions.detach().numpy()
