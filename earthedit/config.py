"""YAML configs, one file per dataset."""
import yaml

SD3_ID = 'stabilityai/stable-diffusion-3-medium-diffusers'


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault('pretrained_model', SD3_ID)
    return cfg
