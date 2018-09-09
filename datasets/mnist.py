import numpy as np

'''
  This module specifies information about a specific dataset. Must include:
    1. ICOLS:       non-data column labels (rowid, ground truth, train/test etc.)
    2. COLS:        data column labels (NOT including ICOLS)
    3. CLASS:       list of classification targets
    4. GROUPS:      class groups
    5. GROUP_NAMES: class group names
    6. GT:          ground truth column index
  Additional data that may be used throught an application should be added.
'''

DIMS = (28,28)
LEN = np.prod(DIMS)
GRID = np.array(range(LEN)).reshape(DIMS)

''' 1. non-data column labels '''
ICOLS = ['class']

''' 2. data column labels '''
COLS = ['pixel %d' % i for i in GRID.flatten()]

''' 3. class labels '''
CLASS = list(range(10))

''' 4. class groups '''
GROUPS = [CLASS] + [[c] for c in CLASS]

''' 5. class group names '''
GROUP_NAMES = ['all'] + list(map(str, CLASS))

''' 8. ground truth column index '''
GT = 0

CHANNELS = [range(DIMS[0]*DIMS[1])]
