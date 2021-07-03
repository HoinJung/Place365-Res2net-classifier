
import os
import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("__file__"))))

import train
from config import parse

# os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
config_path = './param_config.yml'
config = parse(config_path)

# make model output dir
os.makedirs(os.path.dirname(config['result_path']), exist_ok=True)

trainer = train.Trainer(config=config)
trainer.train()
