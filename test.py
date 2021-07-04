import os
import sys
import inferer
from config import parse

os.environ['CUDA_VISIBLE_DEVICES']='1'
config_path = './param_config.yml'
config = parse(config_path)

inferer = inferer.Inferer(config)
inferer()


