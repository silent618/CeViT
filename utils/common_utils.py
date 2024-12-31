from __future__ import print_function, division

import numpy as np
from numpy import *

import torch


def set_seed(self, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    random_state_Kfold = seed


def format_time(seconds):
    """Formats the time in seconds into a string of hours, minutes, and seconds."""
    hours = int(seconds) // 3600
    minutes = (int(seconds) % 3600) // 60
    seconds = int(seconds) % 60
    return f'{hours:02d}h {minutes:02d}m {seconds:02d}s'


def print_training_progress(epoch, total_epochs, time_elapsed, kfold, fold):
    """Prints the training progress including estimated total training time, time used, and estimated time left."""
    estimated_total_time = time_elapsed * ((total_epochs * kfold) / (total_epochs * fold + (epoch + 1)))
    time_used = format_time(time_elapsed)
    time_left = format_time(estimated_total_time - time_elapsed)
    total_training_time = format_time(estimated_total_time)

    print(f'Time used: {time_used: >32}')
    print(f'Estimated time left: {time_left: >22}')
    print(f'Estimated total training time: {total_training_time: >12}')


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )