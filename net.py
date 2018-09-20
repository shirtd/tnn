#!/usr/bin/env python
from src.data import get_data
from src.args import parser
from src.util import sprint, all_stats
from src.cnet import cnet
from src.tda import *
import pickle as pkl
import numpy as np
import sys, os

def main(args):#, masks=[], jdict = {}):
    if args.verbose:
        sprint(0, '[ args ]')
        for k in sorted(args.__dict__.keys()):
            sprint(2, "({}): {}".format(k, args.__dict__[k]))
    # if len(masks) == 0 and not args.no_mask:
    fin = os.path.join(args.sdir, args.data + '.pkl')
    if not args.save and os.path.exists(fin): # and len(args.load) > 0:
        sprint(1, '> loading %s' % fin)
        with open(fin, 'r') as f:
            masks = pkl.load(f)
            # pdict = pkl.load(f)
        # jdicts = pdict['jdicts']
        # masks = pdict['masks']
        # # train, C = get_data(args.data, 'train', args.dir)
        train, C = get_data(args.data, 'train', args.dir)
    else:
        train, C = get_data(args.data, 'train', args.dir)
        jdicts = [build(args, t, i) for i,t in enumerate(train)]
        masks = [np.swapaxes(np.swapaxes(build_mask(jdicts, c), 1, 3), 2, 3) for c in C]
        # masks = map(fmask, [build_mask(jdicts, c) for c in C])
        masks = map(fmask, masks)
        # pdict = {'jdicts' : jdicts, 'masks' : masks}
        if args.save:
            if not os.path.exists(args.sdir):
                sprint(2, '! creating directory %s' % args.sdir)
                os.mkdir(args.sdir)
            fout = os.path.join(args.sdir, args.data + '.pkl')
            sprint(1, '+ writing to %s' % fout)
            with open(fout, 'w') as f:
                # pkl.dump(pdict, f)
                pkl.dump(masks, f)

    # # train, C = get_data(args.data, 'train', args.dir)
    # sys.stdout.write('[ model ')
    # net = cnet(args, masks)
    # return jdicts, masks, net

    stats = zip(*all_stats(train, masks).reshape(-1, 2))
    return masks, stats

if __name__ == '__main__':
    args = parser.parse_args()
    # jdicts, masks, net = main(args)
    masks, stats = main(args)
    net = cnet(args, masks, stats)

# X = torch.stack([m * x for m in masks], 0).view(len(masks), *shape)

# #     # if args.test:
# #     #     # mn = min(map(len, jdict['masks'].values()))
# #     #     mn = min(map(len, masks.values()))
# #     #     l = args.k if args.k < mn else mn
# #     #     _masks = np.array([jdict['masks'][c][:l] for c in jdict['keys']])
# #     # # else:
# #     # #     masks = [fmask(jdict['masks'][c]) for c in jdict['keys']]
# # else:
# #     train = get_data(args.data, 'train', args.dir)
# #     test = get_data(args.data, 'test', args.dir)
# train,C = get_data(args.data, 'train', args.dir)
# test,_C = get_data(args.data, 'test', args.dir)
# jdicts = [build(args, t, i) for i,t in enumerate(train)]
# masks = [build_mask(jdicts, c) for c in C]
