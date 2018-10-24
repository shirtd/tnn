#!/usr/bin/env python
from src.data import get_data
from src.args import parser
from src.util import sprint, all_stats
from src.cnet import cnet
from src.tda import *
import pickle as pkl
import numpy as np
import sys, os

def main(args):
    if args.verbose:
        sprint(0, '[ args ]')
        for k in sorted(args.__dict__.keys()):
            sprint(2, "({}): {}".format(k, args.__dict__[k]))
    # fin = os.path.join(args.sdir, args.data + '.pkl')
    # if not args.save and os.path.exists(fin):
    #     sprint(1, '> loading %s' % fin)
    #     with open(fin, 'r') as f:
    #         masks = pkl.load(f)
    #     train, C = get_data(args.data, 'train')
    # else:
    #     train, C = get_data(args.data, 'train')
    #     jdicts = [build(args, t, i) for i,t in enumerate(train)]
    #     masks = map(fmask, [build_mask(jdicts, c) for c in C])
    #     masks = [np.swapaxes(np.swapaxes(m, 0, 2), 1, 2) for m in masks]
    #     # masks = [np.swapaxes(np.swapaxes(build_mask(jdicts, c), 1, 3), 2, 3) for c in C]
    #     # # masks = map(fmask, [build_mask(jdicts, c) for c in C])
    #     # masks = map(fmask, masks)
    #     if args.save:
    #         if not os.path.exists(args.sdir):
    #             sprint(2, '! creating directory %s' % args.sdir)
    #             os.mkdir(args.sdir)
    #         fout = os.path.join(args.sdir, args.data + '.pkl')
    #         sprint(1, '+ writing to %s' % fout)
    #         with open(fout, 'w') as f:
    #             pkl.dump(masks, f)
    train, C = get_data(args.data, 'train')
    X, y = train[0]['X'] / train[0]['X'].max(), train[0]['y']
    masks = map(sum, [X[filter(lambda i: y[i] == c, range(len(y)))] for c in C])
    sprint(1,'[ calculating stats')
    stats = zip(*all_stats(train, masks).reshape(-1, 2))
    # del train
    return map(lambda x: x.reshape(-1, 28, 28), masks), stats
    # return masks, jdicts

if __name__ == '__main__':
    args = parser.parse_args()
    masks, stats = main(args)
    net = cnet(args, masks, stats)

# X = torch.stack([m * x for m in masks], 0).view(len(masks), *shape)
#
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
