import os

weights_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           'weights')

from . import config, datgaloader, main, model, tester, train


if not os.path.isdir(weights_dir):
    os.mkdir(weights_dir)
