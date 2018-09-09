#!/usr/bin/env python
from functools import partial
from src.data import get_data
from src.util import sprint
from src.args import parser
from src.plot import *
from src.tda import *
import sys, os

def main(args, data='train', dir='plot'):
    if args.verbose:
        sprint(0, '[ args ]')
        for k in sorted(args.__dict__.keys()):
            sprint(2, "({}): {}".format(k, args.__dict__[k]))
    train, C = get_data(args.data, 'train', args.dir)
    jdicts = [build(args, t, i) for i,t in enumerate(train)]
    masks = {c : build_mask(jdicts, c) for c in C}
    plot(args, jdicts, masks)
    return jdicts, masks

if __name__ == '__main__':
    args = parser.parse_args()
    jdicts, masks = main(args)
    # jdicts = main(args)

# sprint(1, '[ getting masks in dimension 0-%d' % args.dims)
# # print(train['shape'])
# # l = train['shape'][1] * train['shape'][2]
# jdicts = [get_persist(t, args.dims) for t in train]
# sprint(1, '[ building masks in dimension %d' % args.dim)
# for jdict in jdicts:
#     jdict['masks'] = get_masks(jdict, args.dim, args.k)
