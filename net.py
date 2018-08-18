from src.args import parser
import src.tda as tda
from src.util import sprint
from src.cnet import cnet
import numpy as np
import sys, os

def main(args, masks=None, jdict = {}):
    sprint(0, '[ args ]')
    for k in sorted(args.__dict__.keys()):
        sprint(2, "({}): {}".format(k, args.__dict__[k]))
    if masks == None and not args.no_mask:
        train = tda.get_data('train', args.dir)
        test = tda.get_data('test', args.dir)
        sprint(1, '[ getting masks in dimension 0-%d' % args.dims)
        jdict = tda.get_masks(train, args.dims)
        sprint(1, '[ building masks in dimension %d' % args.dim)
        jdict['masks'] = tda.build_masks(jdict, args.dim, args.k)
        masks = [tda.fmask(jdict['masks'][c]) for c in jdict['keys']]
    sprint(0, '[ model ]')
    jdict['net'] = cnet(args, masks)
    return jdict

if __name__ == '__main__':
    args = parser.parse_args()
    jdict = main(args)
