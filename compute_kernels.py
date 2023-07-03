import numpy as np
import sklearn
import torch
from sklearn.metrics import pairwise_distances, pairwise_kernels


def gaussian(sqdist, sigma=1):
    K = np.exp(-sqdist / (2 * (sigma * sqdist.max()) ** 2))
    tmp = K.max()
    if tmp == 0:
        tmp = 1
    return K/tmp


def polynomial(Pdot, c=0, d=1):
    return (Pdot + c) ** d

def cosine(Pdot):
    return Pdot / torch.outer(torch.diag(Pdot), torch.diag(Pdot)) ** 0.5
def tanh(Pdot):
    return torch.tanh(Pdot)

def compute_kernels(X):

    Pdot = torch.mm(X, X.T)

    K = cosine(Pdot)
    return K

def compute_kernelst(X):

    Pdot = torch.mm(X, X.T)

    K = tanh(Pdot)
    return K

def compute_kernelsp(X):

    Pdot = torch.mm(X, X.T)

    K = polynomial(Pdot)
    return K


def compute_kernelsg(data, γ=0.5):
	K = sklearn.metrics.pairwise.rbf_kernel(data, gamma=γ)
	return K
