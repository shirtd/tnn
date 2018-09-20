import numpy as np
from src.util import *

# def get_stats(data, axis=0):
#     mean = np.mean(data, axis=axis)
#     std = np.std(data, axis=axis, ddof=0)
#     return {'mean' : mean, 'std' : std}
#
# def standardize(data, stats, axis=0):
#     mean, std = stats['mean'], stats['std']
#     for i,s in enumerate(stats):
#         # x = np.take(data, i, axis)
#         x = data[:,i]
#         # y = (x - mean[i]) / (1. if std[i] == 0 else std[i])
#         # np.put_along_axis(data, i, y, axis)
#         data[:,i] = (x - mean[i]) / (1. if std[i] == 0 else std[i])
#     return data

# n0, n1, c, m, n = 4, 3, 2, 5, 15
# masks = [10*(l+1)*np.array([[[k*(i + j) for i in range(n0)] for j in range(n1)] for k in (-1, 1)]) for l in range(m)]
# M = np.array(masks).reshape(m*c,n0*n1)
# X = np.array([1000*(i+1)*np.ones(n0*n1) for i in range(n)])
# # MX = np.array([[mask * x for mask in M] for x in X])
# MX = np.array([[mask * x for x in X] for mask in M])
# S = [get_stats(x) for x in MX]
# Y = [standardize(MX[i], s) for i,s in enumerate(S)]
