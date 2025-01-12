#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 18:25:28 2025

@author: azad
"""

#!pip3 install pyro-ppl

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import pyro
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
import pyro.distributions.constraints as constraints

import matplotlib.pyplot as plt

from tqdm import trange
from copy import deepcopy
from sys import maxsize