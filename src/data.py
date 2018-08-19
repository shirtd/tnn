from args import DIR
from mnist import MNIST

LOAD = {'train': lambda x: x.load_training(), 'test': lambda x: x.load_testing()}

def get_data(key='train', dir=DIR):
    return dict(zip(('X','y'), LOAD[key](MNIST(dir, return_type='numpy'))))
