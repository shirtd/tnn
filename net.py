from src.data import get_data
from src.args import parser
from src.util import sprint
from src.cnet import cnet
from src.tda import *
import numpy as np
import sys, os

def main(args, masks=[], jdict = {}, l=5, f=lambda x: x):
    sprint(0, '[ args ]')
    for k in sorted(args.__dict__.keys()):
        sprint(2, "({}): {}".format(k, args.__dict__[k]))
    if len(masks) == 0 and not args.no_mask:
        train = get_data('train', args.dir)
        test = get_data('test', args.dir)
        sprint(1, '[ getting masks in dimension 0-%d' % args.dims)
        jdict = get_persist(train, args.dims)
        sprint(1, '[ building masks in dimension %d' % args.dim)
        jdict['masks'] = get_masks(jdict, args.dim, args.k)
        if args.test:
            mn = min(map(len, jdict['masks'].values()))
            l = l if l < mn else mn
            masks = np.array([jdict['masks'][c][:l] for c in jdict['keys']])
            sprint(2, masks.shape)
        else:
            masks = [fmask(jdict['masks'][c]) for c in jdict['keys']]
    sys.stdout.write('[ model ')
    jdict['net'] = cnet(args, masks)
    return jdict

if __name__ == '__main__':
    args = parser.parse_args()
    jdict = main(args)
