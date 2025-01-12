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






class bayesNMF():
    
    def __init__(self, x: torch.Tensor, rank: int, num_steps: int, seed: int=10):
        
        super().__init__()

        self.seed = seed

        # set seeds for PyTorch and Pyro
        torch.manual_seed(self.seed)
        pyro.set_rng_seed(self.seed)

        if isinstance(x, pd.DataFrame):
            self.df = x
            self.x = torch.tensor(x.values, dtype=torch.float32)
        else:
            raise ValueError("Input (x) must be a Pandas DataFrame!")

        self.num_samples = self.x.shape[0]
        self.num_features = self.x.shape[1]
        self.num_signatures = rank
        self.num_steps = num_steps

        self.hyperparameters = {
            "alpha_conc": torch.rand(self.num_signatures),
            "beta_conc": torch.rand(self.num_features)
        }

        self.init_params = {
            "alpha_q" : torch.ones(self.num_samples, self.num_signatures) * 0.1, # Initialize with small positive value
            #"alpha_q" : dist.Dirichlet(self.hyperparameters["alpha_conc"]).sample((self.num_samples, )),
            "beta_q"  : torch.ones(self.num_signatures, self.num_features) * 0.1,  # Initialize with small positive value
            #"beta_q"  : dist.Dirichlet(self.hyperparameters["beta_conc"]).sample((self.num_signatures, ))
        }

        self.adam_params = {"lr": 0.005}

        self.alpha = None
        self.beta = None


    def model(self):

        # set seeds for PyTorch and Pyro
        torch.manual_seed(self.seed)
        pyro.set_rng_seed(self.seed)

        # PRIOR DIST.
        #-----------------------------------------------------------------------
        # exposure matrix   [num_samples X num_signatures]
        with pyro.plate("n1", self.num_samples):
            alpha = pyro.sample("exposures", dist.Dirichlet(self.hyperparameters["alpha_conc"]))

        # signatures matrix [num_signatures X num_features]
        with pyro.plate("k", self.num_signatures):
            beta = pyro.sample("signatures", dist.Dirichlet(self.hyperparameters["beta_conc"]))

        # LIKELIHOOD DIST.
        #-----------------------------------------------------------------------
        # likelihood dist. parameter (lambda) - x, alpha and beta must have same data type to get multiplied
        expectation = torch.matmul(torch.matmul(torch.diag(torch.sum(self.x, axis = 1)), alpha), beta)
        # likelihood dist.
        with pyro.plate("m", self.num_features):
            with pyro.plate("n2", self.num_samples):
                pyro.sample("obs", dist.Poisson(expectation), obs = self.x)


    def guide(self):

        # set seeds for PyTorch and Pyro
        torch.manual_seed(self.seed)
        pyro.set_rng_seed(self.seed)

        # define variational distributions and parameters for exposure matrix
        with pyro.plate("n1", self.num_samples):
            alpha_q = pyro.param("alpha_q", self.init_params["alpha_q"], constraint=constraints.simplex)    # variational parameter
            pyro.sample("exposures", dist.Delta(alpha_q).to_event(1))                                       # variational dist

        # define variational distributions and parameters for signatures matrix
        with pyro.plate("k", self.num_signatures):
            beta_q = pyro.param("beta_q", self.init_params["beta_q"], constraint=constraints.simplex)   # variational parameter
            pyro.sample("signatures", dist.Delta(beta_q).to_event(1))                                   # variational dist


    def _get_params(self):
        """
        retrives the parameters from the param store
        """
        params = dict()
        params["alpha_q"] = pyro.param("alpha_q").detach()
        params["beta_q"] = pyro.param("beta_q").detach()
        return params

    def _get_log_likelihood(self):
        log_likelihood = self._get_log_likelihood_flat()
        return log_likelihood.sum().item()

    def _get_log_likelihood_flat(self):
        params = self._get_params()
        alpha = params["alpha_q"]
        beta = params["beta_q"]
        alpha_hat = torch.sum(self.x, axis=1).unsqueeze(1) * alpha
        rate = torch.matmul(alpha_hat, beta)
        log_likelihood = dist.Poisson(rate).log_prob(self.x)
        return log_likelihood

    def _get_bic(self):
        log_likelihood = self._get_log_likelihood()
        k = (self.num_samples * self.num_signatures) + (self.num_signatures * self.num_features)
        bic = k * torch.log(torch.tensor(self.num_samples, dtype=torch.float64)) - (2 * log_likelihood)
        return bic.item()

    def exposure_to_df(self, save_csv=False, file_name=None):
        """
        returns the exposure matrix as a Pandas DataFrame
        """
        sample_names, contexts = list(self.df.index), list(self.df.columns)
        signames = ["SBS-D"+str(d+1) for d in range(self.num_signatures)]
        alpha = pd.DataFrame(self.alpha, index=sample_names, columns=signames)
        #beta = pd.DataFrame(self.beta_q, index=signames, columns=contexts)

        if save_csv:
            if file_name is None:
                file_name = "exposure.csv"
            elif not isinstance(file_name, str):
                file_name += ".csv"
            elif not file_name.endswith(".csv"):
                file_name += ".csv"
            alpha.to_csv(file_name, index=True, header=True)

        return alpha

    def signatures_to_df(self):
        """
        returns the signatures matrix as a Pandas DataFrame
        """
        sample_names, contexts = list(self.df.index), list(self.df.columns)
        signames = ["SBS-D"+str(d+1) for d in range(self.num_signatures)]
        #alpha = pd.DataFrame(self.alpha_q, index=sample_names, columns=signames)
        beta = pd.DataFrame(self.beta, index=signames, columns=contexts)
        return beta


    def _fit(self):
        """
        fits the model
        """
        pyro.clear_param_store()  # always clear the param store before the inference
        pyro.set_rng_seed(self.seed) # set the seed

        optimizer = pyro.optim.Adam(self.adam_params)

        # set seeds for PyTorch and Pyro
        #torch.manual_seed(self.seed)
        #pyro.set_rng_seed(self.seed)

        svi = SVI(self.model, self.guide, optimizer, loss = Trace_ELBO())

        self.losses, self.log_likelihoods = list(), list()
        t = trange(self.num_steps, desc="Bar desc", leave=True)
        for i in t:  # inference - do gradient steps
            loss = float(svi.step())
            self.losses.append(loss)
            self.log_likelihoods.append(self._get_log_likelihood())
            if i % 5 == 0:
                t.set_description("k=%d, seed=%d  |  ELBO %f" % (self.num_signatures, self.seed, loss))
                t.refresh()

        # store the fitted parameters after optimzation
        params = self._get_params()
        self.alpha = params["alpha_q"]
        self.beta = params["beta_q"]
        self.log_likelihood = self._get_log_likelihood()
        self.bic = self._get_bic()

