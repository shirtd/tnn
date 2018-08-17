from src.args import parser
from src.plot import *
import src.tda as tda
import sys, os

def main(args, data='train'):
    sys.stdout.write('[ args ] ')
    print(args.__dict__)
    train = tda.get_data('train', args.dir)
    print('[ getting masks in dimension 0-%d' % args.dims)
    jdict = tda.get_masks(train, args.dims)
    print('[ building masks in dimension %d' % args.dim)
    jdict['masks'] = tda.build_masks(jdict, args.dim)

    plot_dgms(ax[0], jdict['barcodes'])
    plot_masks(ax[1], jdict['masks'])
    fig[0].savefig(os.path.join('plot','dgm.png'))
    fig[1].savefig(os.path.join('plot','masks.png'))
    return jdict

if __name__ == '__main__':
    args = parser.parse_args()
    jdict = main(args)
