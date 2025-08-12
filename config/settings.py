# from ruamel.yaml import YAML #用来更新yaml配置文件
import argparse
import os

import yaml

from utils.util import get_project_dir


def load_config():
    default_config_file = get_project_dir() + "default.yaml"
    """加载配置文件"""
    parser = argparse.ArgumentParser(description="Server configuration")
    parser.add_argument("--config_path", type=str, default=default_config_file)
    args = parser.parse_args()
    with open(args.config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    master = config['envs']['master']
    for k, v in master.items():
        os.environ[k] = v
    return config['envs']['app']


if __name__ == '__main__':
    print(load_config())
