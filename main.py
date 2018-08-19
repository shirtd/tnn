from functools import partial
from src.util import sprint
from src.args import parser
from src.plot import *
import src.tda as tda
import sys, os

def main(args, data='train', dir='plot'):
    sprint(0, '[ args ]')
    for k in sorted(args.__dict__.keys()):
        sprint(2, "({}): {}".format(k, args.__dict__[k]))
    train = tda.get_data('train', args.dir)
    sprint(1, '[ getting masks in dimension 0-%d' % args.dims)
    jdict = tda.get_persist(train, args.dims)
    sprint(1, '[ building masks in dimension %d' % args.dim)
    jdict['masks'] = tda.get_masks(jdict, args.dim, args.k)

    plot_dgms(ax[0], jdict['barcodes'])
    plot_masks(ax[1], jdict['masks'])
    f = partial(os.path.join, dir)
    fdgm,fmask = map(f, ('dgm.png','masks.png'))
    sprint(2, '| saving %s' % fdgm)
    fig[0].savefig(fdgm)
    sprint(2, '| saving %s' % fmask)
    fig[1].savefig(fmask)
    return jdict

if __name__ == '__main__':
    args = parser.parse_args()
    jdict = main(args)
