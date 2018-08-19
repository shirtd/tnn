import argparse

K = 28
DIMS = 1
DIM = DIMS
PLOT = 'plot'
DATA = 'data'
DIR = 'mnist'
FOUT = DIR

parser = argparse.ArgumentParser(description='mnist persistence.')
parser.add_argument('--dir', default=DIR, help='source data directory. default: %s' % DIR)
parser.add_argument('-D', '--dims', type=int, default=DIMS, help='max dimension. default: %d' % DIMS)
parser.add_argument('-d', '--dim', type=int, default=DIM, help='analysis dimension. default: %d' % DIM)
parser.add_argument('-k', '--k', type=int, default=K, help='number of features to use. default: %s' % K)
parser.add_argument('-b', '--batch', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('-t', '--test-batch', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
parser.add_argument('-e', '--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('-l', '--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--log', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--no-mask', action='store_true', default=False, help='no masks')
parser.add_argument('--test', action='store_true', default=False, help='test 3d convolution')
parser.add_argument('-v','--verbose', action='store_true', default=False, help='verbose train')
parser.add_argument('--plot', default='dgm', help='persistence representation. default: diagram')
parser.add_argument('--save', action='store_true', help='save plots')
parser.add_argument('--pdir', default=PLOT, help='plot directory')
parser.add_argument('--data', default=DATA, help='data directory')
parser.add_argument('--load', default=FOUT, help='file to load. default: %s' % FOUT)
parser.add_argument('--fout', default=FOUT, help='file to save. default: %s' % FOUT)
# parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
# parser.add_argument('-t', '--transpose', action='store_false', help='sample space persistence (no transpose)')
# parser.add_argument('-','--dims', type=int, nargs='+', default=[0,1], help='')
