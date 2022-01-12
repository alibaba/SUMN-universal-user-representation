from argparse import Namespace
# import tensorflow as tf
from tensorflow.python.platform import gfile
import json


_ARCH_CONF_FILE_NAME = "model_arch.json"

def _get_arch_conf_file_path(dir_name):
    file_path = dir_name
    if not file_path.endswith("/"):
        file_path += "/"

    return file_path + _ARCH_CONF_FILE_NAME


def dump_args(path, args_obj):
    path = _get_arch_conf_file_path(path)
    with gfile.GFile(path, "w") as writer:
        json.dump(
            args_obj.__dict__,
            writer,
            ensure_ascii=False
        )


def load_args(path):
    path = _get_arch_conf_file_path(path)
    args = Namespace()
    with gfile.GFile(path, "r") as reader:
        obj = json.load(reader)
        for key, value in obj.items():
            args.__setattr__(key, value)

    args.__dict__.setdefault("pooling", "max")
    args.__dict__.setdefault("cell", "gru")
    args.__dict__.setdefault("num_gru_heads", 1)

    return args
