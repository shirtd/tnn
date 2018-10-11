from __future__ import print_function
import torch, sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.util import sprint
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, CIFAR10
from sklearn.metrics import *
import pandas as pd

DATASETS = {'mnist' : MNIST, 'cifar' : CIFAR10}
SHAPE = {'mnist' : (1, 28, 28), 'cifar' : (3, 32, 32)}
CLASS = {'mnist' : 10, 'cifar' : 10}

''' UTIL '''
def pad(k):
    return int(np.floor(float(k) / 2))

def nbr_expand(x, d, s):
    i, j, k = s
    return x.view(i, j*k)[:, d].view(i, j, k, -1)

''' DEFAULT '''
class MaskTensor(object):
    def __init__(self, d, shape):
        super(MaskTensor, self).__init__()
        self.shape = shape
        self.n = len(d)
        self.d = d
    def __call__(self, x):
        return torch.cat([nbr_expand(x, d, self.shape) for d in self.d], 0)

class Net(nn.Module):
    def __init__(self, masks, s):
        super(Net, self).__init__()
        ''' in/out '''
        self.k = 10
        self.masks = masks
        self.c, self.w1, self.w2 = s
        self.n0 = self.c * (len(self.masks) if len(self.masks) > 0 else 1)
        self.n = len(masks)
        ''' neurons '''
        # convolution
        self.n1 = self.n0 * 2
        self.n2 = self.n0 * 4
        self.k1, self.k2 = 4, 4
        self.s1, self.s2 = 1, 1
        self.p1,self.p2 = self.k1 / 2, self.k2 / 2
        self.x1 = self.w1 / (self.n1/self.n0) / (self.n2/self.n1)
        self.x2 = self.w2 / (self.n1/self.n0) / (self.n2/self.n1)
        # connected
        # self.n3 = self.n0 * 64 #/ (self.s1 * self.s2)
        # self.n4 = self.n0 * 32 #/ (self.s1 * self.s2)
        # self.n5 = self.n0 * 16 #/ (self.s1 * self.s2)
        # self.n6 = self.n0 * 8
        # self.n7 = self.n0 * 4
        # self.n8 = self.n0 * 2
        self.n3 = self.n2 * self.x1 * self.x2
        self.n4 = self.n3 / 2
        self.n5 = self.n4 / 2
        self.n6 = self.n5 / 2
        self.n7 = self.n6 / 2
        self.n8 = self.n7 / 2
        self.n9 = self.n8 / 2

        ''' layers '''
        # convolution
        self.conv0 = nn.Conv3d(self.n0, self.n0, (1, 1, self.k))
        self.conv1 = nn.Conv2d(self.n0, self.n1, self.k1, self.s1, self.p1)
        self.conv2 = nn.Conv2d(self.n1, self.n2, self.k2, self.s2, self.p2)
        self.conv2_drop = nn.Dropout2d()
        # connected
        self.fc1 = nn.Linear(self.n3, self.n4)
        self.fc2 = nn.Linear(self.n4, self.n5)
        self.fc3 = nn.Linear(self.n5, self.n6)
        self.fc4 = nn.Linear(self.n6, self.n7)
        self.fc5 = nn.Linear(self.n7, self.n8)
        self.fc6 = nn.Linear(self.n8, self.n9)
        self.fc7 = nn.Linear(self.n9, self.n)

    def view(self, x):
        return x.view(-1, self.n3)

    def forward(self, x):
        x = self.conv0(x)
        x = x.view(-1, self.n0, self.w1, self.w2)
        ''' convolution '''
        # in -> conv1
        # print(x.shape)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        # conv1 -> conv2
        # print(x.shape)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        # print(x.shape)

        ''' connected '''
        x = self.view(x)
        # print(x.shape)
        # conv2 -> linear1
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # linear1 -> linear2
        x = self.fc2(x)
        x = F.relu(x)
        # linear2 -> linear3
        x = self.fc3(x)
        x = F.relu(x)
        # linear3 -> linear4
        x = self.fc4(x)
        x = F.relu(x)
        # linear4 -> linear5
        x = self.fc5(x)
        x = F.relu(x)
        # linear5 -> linear6
        x = self.fc6(x)
        x = F.relu(x)
        # linear6 -> linear7 (out)
        x = self.fc7(x)
        return F.log_softmax(x, dim=1)

''' RUN TRAIN '''
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    if args.verbose:
        sprint(1, '[ %d train' % epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if args.verbose:
            if batch_idx % args.log == 0:
                sprint(2, '| {:.0f}%\tloss:\t{:.6f}'.format(
                    100. * batch_idx / len(train_loader),
                    loss.item()))

''' RUN TEST '''
def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    dfp, dfl = pd.DataFrame(), pd.DataFrame()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            dfp = dfp.append(pd.DataFrame(F.softmax(output,dim=1).tolist()), sort=False, ignore_index=True)
            dfl = dfl.append(pd.DataFrame(target.tolist()), sort=False, ignore_index=True)

    y = [l[0] for l in dfl.values]
    test_loss /= len(test_loader.dataset)
    accuracy = float(100. * correct) / float(len(test_loader.dataset))
    score = 100 / (1 + log_loss(y, dfp.values, eps=1E-15))
    sprint(1, '[ {}\ttest\t{:.5f}\t{:.4f}\t{:.2f}%'.format(epoch, test_loss, score, accuracy))
    return test_loss

''' RUN '''
def mnet(args, masks, k):# stats):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    DATA = DATASETS[args.data]
    shape = SHAPE[args.data]

    train_data = DATA('../data', train=True, download=True,
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                        MaskTensor(masks, shape)])),
                        # transforms.Normalize(*stats)])),
    test_data = DATA('../data', train=False,
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                        MaskTensor(masks, shape)]))
                        # transforms.Normalize(*stats)])),

    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, **kwargs)
    test_loader = DataLoader(test_data, batch_size=args.test_batch, shuffle=True, **kwargs)

    # print('raw data shape')
    # print(train_loader.dataset.train_data.shape)
    model = Net(masks, shape).to(device)

    print(str(model)[:-2])

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                                factor=0.05, cooldown=5, patience=10)

    print('[epoch\tmode\tloss\tscore\taccuracy')
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step(test(args, model, device, test_loader, epoch))

    return model

# ''' TEST '''
# class TestTensor(object):
#     def __init__(self, masks, shape):
#         super(TestTensor, self).__init__()
#         self.shape = shape
#         self.masks = torch.from_numpy(masks).float()
#     def __call__(self, sample):
#         if len(self.masks) == 0:
#             return sample
#         x = sample.view(*shape)
#         X = torch.stack([m * x for m in self.masks], 0)
#         return X
#
# class TestNet(nn.Module):
#     def __init__(self, masks, k,  n, dim):
#         super(TestNet, self).__init__()
#         ''' in/out '''
#         self.k = k
#         self.dim = dim
#         self.n0 = 1 if len(masks) == 0 else len(masks)
#         self.n = n
#         ''' neurons '''
#         # convolution
#         self.n1 = self.n0 * 10
#         self.n2 = self.n0 * 20
#         self.k1, self.k2 = 4, 4
#         self.s1, self.s2 = 2, 2
#         self.p1 = self.k1 / 2
#         self.p2 = self.k2 / 2
#         self.r1, self.r2 = 2, 2
#
#         # connected
#         self.n3 = self.n0 * 160 * dim / k
#         self.n4 = self.n0 * 110 * dim / k
#         self.n5 = self.n0 * 50 * dim / k
#
#         ''' layers '''
#         # convolution
#         self.conv1 = nn.Conv3d(self.n0, self.n1, self.k1, self.s1, self.p1, groups=self.n0)
#         self.conv2 = nn.Conv3d(self.n1, self.n2, self.k2, self.s2, self.p2)#, groups=self.n0)
#         self.conv2_drop = nn.Dropout3d()
#         # connected
#         self.fc1 = nn.Linear(self.n3, self.n4)
#         self.fc2 = nn.Linear(self.n4, self.n5)
#         self.fc3 = nn.Linear(self.n5, self.n)
#
#     def view(self, x):
#         return x.view(-1, self.n3)
#
#     def forward(self, x):
#         ''' convolution '''
#         # in -> conv1
#         # print(x.shape)
#         x = self.conv1(x)
#         # print(x.shape)
#         x = F.max_pool3d(x, self.r1)
#         # print(x.shape)
#         x = F.relu(x)
#         # print()
#
#         # conv1 -> conv2
#         x = self.conv2(x)
#         # print(x.shape)
#         x = self.conv2_drop(x)
#         x = F.max_pool3d(x, self.r2)
#         # print(x.shape)
#         x = F.relu(x)
#         # print()
#
#
#         ''' connected '''
#         x = self.view(x)
#         # print(x.shape)
#         # conv2 -> linear1
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         # linear1 -> linear2
#         x = self.fc2(x)
#         x = F.relu(x)
#         # linear2 -> linear3 (out)
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)
#
# # def get_load():
# #     return DataLoader(MNIST('../data', train=True, download=True,
# #                 transform = transforms.Compose([transforms.ToTensor()])),
# #             batch_size=1000, shuffle=False)
# #
# # def aload(load=get_load()):
# #     p, l = [], []
# #     for data, target in load:
# #         p.append(data)
# #         l.append(target)
# #     p = torch.cat(p, 0).view(-1,28,28)
# #     return p, torch.cat(l)
