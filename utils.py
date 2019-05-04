from __future__ import print_function
import numpy as np
import torch
import torch.cuda.comm as comm
from torch.nn.init import kaiming_normal_
import torch.nn.functional as F
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
from torch.utils.data.sampler import Sampler
from functools import partial
from nested_dict import nested_dict
from collections import OrderedDict
from random import shuffle


def cast(params, dtype='float', device='cpu'):
    if isinstance(params, dict):
        return {k: cast(v, dtype, device) for k, v in params.items()}
    else:
        return getattr(params, dtype)().to(device)


def conv_params(ni, no, k=1, device='cpu'):
    w = torch.Tensor(no, ni, k, k).to(device).requires_grad_()
    return {'weight': kaiming_normal_(w, mode='fan_out', nonlinearity='relu'),
            'bias': None}


def linear_params(ni, no, device='cpu'):
    w = torch.Tensor(no, ni).to(device).requires_grad_()
    return {'weight': kaiming_normal_(w, mode='fan_out', nonlinearity='relu'),
            'bias': torch.zeros(no).to(device).requires_grad_()}


def bnparams(n, device='cpu'):
    return {'weight': torch.rand(n).to(device).requires_grad_(),
            'bias': torch.zeros(n).to(device).requires_grad_()}


def bnstats(n, device='cpu'):
    return {'running_mean': torch.zeros(n).to(device),
            'running_var': torch.ones(n).to(device)}


def flatten_params(params, device='cpu'):
    return OrderedDict(('.'.join(k), v.to(device).requires_grad_())
                       for k, v in nested_dict(params).iteritems_flat() if v is not None)


def flatten_stats(stats, device='cpu'):
    return OrderedDict(('.'.join(k), v.to(device))
                       for k, v in nested_dict(stats).iteritems_flat())


def batch_norm(x, params, stats, base, mode):
    return F.batch_norm(x, weight=params[base]['weight'],
                        bias=params[base]['bias'],
                        running_mean=stats[base]['running_mean'],
                        running_var=stats[base]['running_var'],
                        training=mode)


def print_tensor_dict(params):
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(params.items()):
        print(str(i).ljust(5), key.ljust(kmax + 3),
              str(tuple(v.size())).ljust(23), v.dtype)


class InfiniteDataLoader(object):
    """Allow to load sample infinitely"""

    def __init__(self, dataloader):
        super(InfiniteDataLoader, self).__init__()
        self.dataloader = dataloader
        self.data_iter = None

    def next(self):
        try:
            data = self.data_iter.next()
        except Exception:
            # Reached end of the dataset
            self.data_iter = iter(self.dataloader)
            data = self.data_iter.next()
            #print('*** infi_data_loader starts over.')
        return data

    def __next__(self):
        return self.next()

    def __len__(self):
        return len(self.dataloader)


class MaskedRandomSampler(Sampler):
     """
     Samples elements from selected indices,
     sampling is repeated s.t. size of one epoch is equal to
     vanilla sampling.
     """
     def __init__(self, mask, rep_num, do_shuffle=True):
         self.mask = mask
         self.rep_num = rep_num
         self.do_shuffle = do_shuffle

     def __iter__(self):
         randperms = []
         for _ in range(self.rep_num):
             if self.do_shuffle:
                 shuffle(self.mask)
             randperms.extend(self.mask)
         return iter(randperms)

     def __len__(self):
         return len(self.mask)*self.rep_num


def create_semisup_sampler(labels, num_classes, num_samples_per_class, do_shuffle=True):
     mask = []
     for c in range(num_classes):
         indices = np.where(np.array(labels)==c)[0].tolist()
         mask.extend(indices[:num_samples_per_class])

     rep_num = max(1, int(len(labels) / num_classes / num_samples_per_class))
     sampler = MaskedRandomSampler(mask, rep_num, do_shuffle=do_shuffle)
     return sampler


class NormalRho(object):
    # rho version
    def __init__(self, mu, rho):
        super(NormalRho, self).__init__()
        self.mu = mu
        self.rho = rho
        self.eps = torch.zeros_like(self.mu, requires_grad=False)
        self.eps = self.eps.to(self.mu.device)

    def logpdf(self, x):
        c = - float(0.5 * math.log(2 * math.pi))
        std = (1 + self.rho.exp()).log()
        logvar = std.pow(2).log()
        return c - 0.5 * logvar - (x - self.mu).pow(2) / (2 * std.pow(2))

    def pdf(self, x):
        return torch.exp(self.logpdf(x))

    def sample(self, training=True):
        if training:
            self.eps.normal_()
        else:
            self.eps.zero_()
        std = (1 + self.rho.exp()).log()
        return self.mu + std * self.eps

    def KL(self, mean_prior, std_prior, do_bayes=False):
        mean, std = self.mu, (1 + self.rho.exp()).log()
        assert(mean.shape == mean_prior.shape[1:])
        K = mean_prior.shape[0]
        mean = mean.expand_as(mean_prior)
        mean, mean_prior = mean.reshape(K, -1), mean_prior.reshape(K, -1)

        if do_bayes:
            std = std.expand_as(std_prior)
            std, std_prior = std.reshape(K, -1), std_prior.reshape(K, -1)

            loss_k = (-(std / std_prior).log() +
                      (std.pow(2) + (mean - mean_prior).pow(2)) / (2 * std_prior.pow(2))
                      - 1 / 2).sum(1)
        else:
            loss_k = (mean - mean_prior).pow(2).sum(1) / 2
        return loss_k

    def reverseKL(self, mean_post, std_post, do_bayes=False):
        mean, std = self.mu, (1 + self.rho.exp()).log()
        assert(mean_post.shape == mean.shape[1:])
        K = mean.shape[0]
        mean_post = mean_post.expand_as(mean)
        mean_post, mean = mean_post.reshape(K, -1), mean.reshape(K, -1)

        if do_bayes:
            std_post = std_post.expand_as(std)
            std_post, std = std_post.reshape(K, -1), std.reshape(K, -1)

            loss_k = (-(std_post / std).log() +
                      (std_post.pow(2) + (mean_post - mean).pow(2)) / (2 * std.pow(2))
                      - 1 / 2).sum(1)
        else:
            loss_k = (mean_post - mean).pow(2).sum(1) / 2
        return loss_k
