# import torch
# from torchvision import datasets, transforms
from functools import partial
from src.data import get_data
import matplotlib.pyplot as plt
from src.tda import *
from src.args import parser
from src.mnet import mnet
import sys, os

args = parser.parse_args()

plt.ion()

def imap(i):
    return (i // 28, i % 28)

def as_mask(c, t):
    m = np.zeros(784, float)
    m[c[:,:2]] = t
    return m

def one_mask(x):
    R = ripser(x.T, do_cocycles=True)
    dgm = R['dgms'][1]
    C = R['cocycles'][1]
    T = dgm[:,1] - dgm[:,0]
    I = sorted(range(len(dgm)), key=lambda i: T[i], reverse=True)
    dgm, T, C = dgm[I], T[I], [C[i] for i in I]
    return sum([as_mask(c,t) for c,t in zip(C,T)])

def one_class(c=0):
    train, classes = get_data('mnist', 'train')
    X, y = train[0]['X'], train[0]['y']
    shape = train[0]['shape']
    I = filter(lambda i: y[i] == c, range(len(y)))
    return one_mask(X[I])

def get_nns(M):
    n = len(M)
    D = np.empty((n, n), int)
    for i in range(n):
        l = [j for j in range(n) if j != i]
        D[i] = [i] + sorted(l, key=lambda x: abs(M[i] - M[x]))
    return D

def class_nns(c=0, k=10):
    M = one_class(c)
    return get_nns(M)[:, :k]

print('computing neighbors')
D = [class_nns(c) for c in range(10)]

net = mnet(args, D, 10)

# print('loading dataset')
# train = datasets.MNIST('../data', train=True, download=True)
# x = transforms.ToTensor()(train[1][0])
# X = torch.cat([nbr_expand(x, d) for d in D], 0)
