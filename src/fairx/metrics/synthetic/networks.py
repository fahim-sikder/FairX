# Original work Copyright 2022, Synthcity Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified work Copyright (c) 2024 Md Fahim Sikder
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

# future
from __future__ import absolute_import, division, print_function

# stdlib
from typing import Any

# third party
import torch
from torch import nn

torch.manual_seed(1)

# Global variables

ACTIVATION_DICT = {
    "ReLU": torch.nn.ReLU(),
    "Hardtanh": torch.nn.Hardtanh(),
    "ReLU6": torch.nn.ReLU6(),
    "Sigmoid": torch.nn.Sigmoid(),
    "Tanh": torch.nn.Tanh(),
    "ELU": torch.nn.ELU(),
    "CELU": torch.nn.CELU(),
    "SELU": torch.nn.SELU(),
    "GLU": torch.nn.GLU(),
    "LeakyReLU": torch.nn.LeakyReLU(),
    "LogSigmoid": torch.nn.LogSigmoid(),
    "Softplus": torch.nn.Softplus(),
}


def build_network(network_name: str, params: dict) -> Any:

    if network_name == "feedforward":

        net = feedforward_network(params)

    return net


def feedforward_network(params: dict) -> Any:

    """Architecture for a Feedforward Neural Network

    Args:

        ::params::

        ::params["input_dim"]::
        ::params[""rep_dim""]::
        ::params["num_hidden"]::
        ::params["activation"]::
        ::params["num_layers"]::
        ::params["dropout_prob"]::
        ::params["dropout_active"]::
        ::params["LossFn"]::

    Returns:

        ::_architecture::

    """

    modules = []

    if params["dropout_active"]:

        modules.append(torch.nn.Dropout(p=params["dropout_prob"]))

    # Input layer

    modules.append(
        torch.nn.Linear(params["input_dim"], params["num_hidden"], bias=False)
    )
    modules.append(ACTIVATION_DICT[params["activation"]])

    # Intermediate layers

    for u in range(params["num_layers"] - 1):

        if params["dropout_active"]:

            modules.append(torch.nn.Dropout(p=params["dropout_prob"]))

        modules.append(
            torch.nn.Linear(params["num_hidden"], params["num_hidden"], bias=False)
        )
        modules.append(ACTIVATION_DICT[params["activation"]])

    # Output layer

    modules.append(torch.nn.Linear(params["num_hidden"], params["rep_dim"], bias=False))

    _architecture = nn.Sequential(*modules)

    return _architecture