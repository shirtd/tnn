from functools import partial
from src.data import get_data
from src.util import sprint
from src.args import parser
from src.plot import *
from src.tda import *
import sys, os

def main(args, data='train', dir='plot'):
    sprint(0, '[ args ]')
    for k in sorted(args.__dict__.keys()):
        sprint(2, "({}): {}".format(k, args.__dict__[k]))
    train = get_data('train', args.dir)
    sprint(1, '[ getting masks in dimension 0-%d' % args.dims)
    jdict = get_persist(train, args.dims)
    sprint(1, '[ building masks in dimension %d' % args.dim)
    jdict['masks'] = get_masks(jdict, args.dim, args.k)
    plot(args, jdict)
    return jdict

if __name__ == '__main__':
    args = parser.parse_args()
    jdict = main(args)
