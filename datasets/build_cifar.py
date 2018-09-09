from cifar import *
import cPickle, os, sys
import pandas as pd

def sprint(i, s):
    print(i*' ' + s)

def unpickle(f):
    with open(f, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

# IN
BASE, TEST = 'data_batch_', 'test_batch'
DIR, SDIR = 'datasets', 'cifar-10'
PATH = os.path.join(DIR, SDIR)

# OUT
DEST, PRE = PATH, 'cifar'

# TRAIN
sprint(1, '[ loading %s*' % os.path.join(PATH, BASE))
df = pd.DataFrame(columns=ICOLS+COLS)
for i in range(1,6):
    FILE = os.path.join(PATH,'%s%d' % (BASE,i))
    sprint(2,'| loading %s' % FILE)
    pkl = unpickle(FILE)
    X, y = pkl['data'], np.array(pkl['labels'])
    df = pd.concat([df, pd.DataFrame(np.vstack([y, X.T]).T, columns=ICOLS+COLS)])

FOUT = os.path.join(DEST, '_'.join([PRE, 'train.csv']))
sprint(1, '+ saving %s' % FOUT)
df.to_csv(FOUT, index=False)

# TEST
FILE = os.path.join(PATH, TEST)
sprint(1, '[ loading %s' % FILE)
pkl = unpickle(FILE)
X, y = pkl['data'], np.array(pkl['labels'])
df = pd.DataFrame(np.vstack([y, X.T]).T, columns=ICOLS+COLS)

FOUT = os.path.join(DEST, '_'.join([PRE, 'test.csv']))
sprint(1, '+ saving %s' % FOUT)
df.to_csv(FOUT, index=False)
