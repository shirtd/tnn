from __future__ import print_function
import torch, sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.util import sprint
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import MNIST

def pad(k):
    return int(np.ceil(float(k) / 2))

class MaskTensor(object):#transforms.ToTensor):
    def __init__(self, masks):
        super(MaskTensor, self).__init__()
        if masks != None:
            self.masks = [torch.from_numpy(m).float() for m in masks]
        else:
            self.masks = None
    def __call__(self, sample):
        if self.masks == None:
            return sample
        x = sample.view(28, 28)
        X = torch.stack([m * x for m in self.masks], 0).view(len(self.masks), 28, 28)
        # y = torch.from_numpy(np.array(labels, dtype=np.int64))
        return X#, y

class TestTensor(object):#transforms.ToTensor):
    def __init__(self, masks):
        super(TestTensor, self).__init__()
        # self.masks = torch.from_numpy(masks).float()
        self.masks = torch.from_numpy(masks).float()
        # if len(masks) > 0:
        # else:
        #     self.masks =
    def __call__(self, sample):
        # if self.masks == None:
        if len(self.masks) == 0:
            return sample
        x = sample.view(28, 28)
        X = torch.stack([m * x for m in self.masks], 0)
        # X = X.view(10, -1, 28, 28)
        # print(X.shape)
        # y = torch.from_numpy(np.array(labels, dtype=np.int64))
        return X#, y

class TestNet(nn.Module):
    def __init__(self, masks, n=10):
        super(TestNet, self).__init__()
        ''' in/out '''
        self.n0 = 1 if len(masks) == 0 else len(masks)
        self.n = n
        ''' neurons '''
        # convolution
        self.n1 = self.n0 * 10
        self.n2 = self.n0 * 20
        self.k1, self.k2 = 5, 5
        self.s1, self.s2 = 1, 1
        self.p1 = self.k1 - 1
        self.p2 = self.k2 - 1

        # connected
        self.n3 = self.n0 * 320 * 22
        self.n4 = self.n0 * 50 * 11
        self.n5 = self.n0 * 10 * 3

        ''' layers '''
        # convolution
        self.conv1 = nn.Conv3d(self.n0, self.n1, self.k1, self.s1, self.p1)
        self.conv2 = nn.Conv3d(self.n1, self.n2, self.k2, self.s2, self.p2)
        self.conv2_drop = nn.Dropout3d()
        # connected
        self.fc1 = nn.Linear(self.n3, self.n4)
        self.fc2 = nn.Linear(self.n4, self.n5)
        self.fc3 = nn.Linear(self.n5, self.n)

    def view(self, x):
        return x.view(-1, self.n3)

    def forward(self, x):
        ''' convolution '''
        # in -> conv1
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = F.max_pool3d(x, 2)
        print(x.shape)
        x = F.relu(x)
        # conv1 -> conv2
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool3d(x, 2)
        x = F.relu(x)

        ''' connected '''
        x = self.view(x)
        # conv2 -> linear1
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # linear1 -> linear2
        x = self.fc2(x)
        x = F.relu(x)
        # linear2 -> linear3 (out)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class Net(nn.Module):
    def __init__(self, masks, n=10):
        super(Net, self).__init__()
        ''' in/out '''
        self.n0 = 1 if masks == None else len(masks)
        self.n = n
        ''' neurons '''
        # convolution
        self.n1 = self.n0 * 10
        self.n2 = self.n0 * 20
        self.k1, self.k2 = 5, 5
        # connected
        self.n3 = self.n0 * 320
        self.n4 = self.n0 * 50
        self.n5 = self.n0 * 10

        ''' layers '''
        # convolution
        self.conv1 = nn.Conv2d(self.n0, self.n1, self.k1)
        self.conv2 = nn.Conv2d(self.n1, self.n2, self.k2)
        self.conv2_drop = nn.Dropout2d()
        # connected
        self.fc1 = nn.Linear(self.n3, self.n4)
        self.fc2 = nn.Linear(self.n4, self.n5)
        self.fc3 = nn.Linear(self.n5, self.n)

    def view(self, x):
        return x.view(-1, self.n3)

    def forward(self, x):
        ''' convolution '''
        # in -> conv1
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        # conv1 -> conv2
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        ''' connected '''
        x = self.view(x)
        # conv2 -> linear1
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # linear1 -> linear2
        x = self.fc2(x)
        x = F.relu(x)
        # linear2 -> linear3 (out)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

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

def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = float(100 * correct) / float(len(test_loader.dataset))
    sprint(1, '[ {}\ttest\t{:.3f}\t{:.6f}'.format(epoch, 100./(1+test_loss), accuracy))
    # sprint(2, '| {:.3f}\t{:.3f}%'.format())
    return test_loss

def cnet(args, masks):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(MNIST('../data', train=True, download=True,
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                        TestTensor(masks) if args.test else MaskTensor(masks)]
                    )), batch_size=args.batch, shuffle=True, **kwargs)
    test_loader = DataLoader(MNIST('../data', train=False,
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                        TestTensor(masks) if args.test else MaskTensor(masks)]
                    )), batch_size=args.test_batch, shuffle=True, **kwargs)

    model = TestNet(masks).to(device) if args.test else Net(masks).to(device)
    print(str(model)[:-2])

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                    'min', verbose=True, cooldown=5, patience=5)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step(test(args, model, device, test_loader, epoch))

    return model

# class MaskDataset(torch.utils.data.Dataset):
#     def __init__(self, X, y, masks, stats=None):
#         self.data, self.gt = X, y
#         self.masks = masks
#         if stats != None:
#             self.mean, self.std  = stats
#         else:
#             sprint(2, '| calculating mean and deviation')
#             self.mean = np.mean(self.data, axis=0)
#             self.std = np.std(self.data, axis=0, ddof=0)
#         self.data = (self.data - self.mean) / self.std
#         self.trans = MaskTensor()#self.masks)
#
#     def __getitem__(self, idx):
#         return self.trans({'data' : self.data[idx], 'labels' : self.gt[idx]})
#
#     def __len__(self):
#         return len(self.gt)
