from src.data import get_data
from src.args import parser
from src.util import sprint
from src.cnet import cnet
from src.tda import *
import pickle as pkl
import numpy as np
import sys, os

def main(args, masks=[], jdict = {}):
    if args.verbose:
        sprint(0, '[ args ]')
        for k in sorted(args.__dict__.keys()):
            sprint(2, "({}): {}".format(k, args.__dict__[k]))
    if len(masks) == 0 and not args.no_mask:
        # fin = os.path.join(args.data, args.load + '%d.pkl' % args.k)
        fin = os.path.join(args.data, args.load + '.pkl')
        if not args.save and os.path.exists(fin) and len(args.load) > 0:
            sprint(1, '[ loading %s' % fin)
            with open(fin, 'r') as f:
                jdict = pkl.load(f)
        else:
            train = get_data('train', args.dir)
            test = get_data('test', args.dir)
            sprint(1, '[ getting masks in dimension 0:%d' % args.dims)
            jdict = get_persist(train, args.dims)
            sprint(1, '[ building masks in dimension %d' % args.dim)
            jdict['masks'] = get_masks(jdict, args.dim, args.k)
            if args.save:
                if not os.path.exists(args.data):
                    sprint(2, '! creating directory %s' % args.data)
                    os.mkdir(args.data)
                # fout = os.path.join(args.data, args.fout + '%d.pkl' % args.k)
                fout = os.path.join(args.data, args.fout + '.pkl')
                sprint(2, '| writing to %s' % fout)
                with open(fout, 'w') as f:
                    pkl.dump(jdict, f)
        if args.test:
            mn = min(map(len, jdict['masks'].values()))
            l = args.k if args.k < mn else mn
            masks = np.array([jdict['masks'][c][:l] for c in jdict['keys']])
            # print(masks.shape)
        else:
            masks = [fmask(jdict['masks'][c]) for c in jdict['keys']]
    else:
        train = get_data('train', args.dir)
        test = get_data('test', args.dir)
    sys.stdout.write('[ model ')
    jdict['net'] = cnet(args, masks)
    return jdict

if __name__ == '__main__':
    args = parser.parse_args()
    jdict = main(args)
