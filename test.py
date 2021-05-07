
import os
import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("__file__"))))

import inferer
from config import parse

os.environ['CUDA_VISIBLE_DEVICES']='1'
# os.environ['CUDA_VISIBLE_DEVICES']='0'
config_path = './param_config.yml'
config = parse(config_path)

inferer = inferer.Inferer(config)
inferer()


