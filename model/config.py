# -*- coding: utf-8 -*-#
from collections import namedtuple


# training configurations
TrainConfigs = namedtuple("TrainConfigs", (
    "learning_rate",
    "batch_size",
    # checkpoint for parameter initialization
    "init_checkpoint",
    "max_grad_norm",
    "weighted_target"
))


# prediction configurations
PredictConfigs = namedtuple("PredictConfigs", (
    "separator",
    "hop"
))


RunConfigs = namedtuple("RunConfigs", (
    "log_every"
))