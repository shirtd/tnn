import argparse

K = -1
DIMS = 1
DIM = DIMS
DIR = 'mnist'

parser = argparse.ArgumentParser(description='mnist persistence.')
parser.add_argument('-D', '--dims', type=int, default=DIMS, help='max dimension. default: %d' % DIMS)
parser.add_argument('-d', '--dim', type=int, default=DIM, help='analysis dimension. default: %d' % DIM)
parser.add_argument('--dir', default=DIR, help='source data directory. default: %s' % DIR)
parser.add_argument('-k','--k', type=int, default=K, help='number of features to use. default: %s' % K)
# parser.add_argument('-t', '--transpose', action='store_false', help='sample space persistence (no transpose)')
# parser.add_argument('-','--dims', type=int, nargs='+', default=[0,1], help='')
