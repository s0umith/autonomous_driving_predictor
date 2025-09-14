import os
import yaml
import easydict


def load_config_act(path):
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return easydict.EasyDict(cfg)


def load_config_init(path):
    path = os.path.join('init/configs', f'{path}.yaml')
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return easydict.EasyDict(cfg)
