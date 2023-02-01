import torch
import math
import torch.nn.functional as F
import numpy as np

C = - 0.5 * math.log( 2 * math.pi)
def KLD(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), -1)
def bernoulli(x_hat, x):
    EPS = 1e-8
    return -(torch.log(x_hat + EPS) * x+ torch.log(1 - x_hat + EPS) * (1-x)).sum(-1).sum(-1)
def gaussian(x, mu, logvar):
    return C - 0.5 * (logvar + torch.square(x - mu) / torch.exp(logvar) )
def std_gaussian(x):
    return C - x**2 / 2
def gaussian_std_margin(mu, logvar):
    return C - 0.5*(torch.square(mu) + torch.exp(logvar))
def gaussian_margin(logvar):
    return C - 0.5*(1 + logvar)
def mse(x, recon_x):
    return torch.nn.functional.mse_loss(recon_x, x) 
def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)
def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor

def matrix_poly(matrix, d):
    x = torch.eye(d).cuda()+ torch.div(matrix, d)
    return torch.matrix_power(x, d)