import numpy as np
import math


def pad_to_length(x, target_length, pad_value):
    assert x.ndim == 1
    return np.pad(
        x, (0, target_length - x.shape[0]), mode='constant', constant_values=pad_value)


def prepare_stop_target(inputs, out_steps):
    max_len = max((x.shape[0] for x in inputs))  # no padding for the longest sequence
    target_len = math.ceil(max_len / out_steps) * out_steps
    return np.stack(
        [pad_to_length(x, target_len, pad_value=1.) for x in inputs])


def prepare_char(inputs, pad_value):
    max_len = max((x.shape[0] for x in inputs))  # no padding for the longest sequence
    return np.stack(
        [pad_to_length(x, max_len, pad_value=pad_value) for x in inputs])


def prepare_phone(inputs, out_steps, pad_value):
    max_len = max((x.shape[0] for x in inputs))  # no padding for the longest sequence
    target_len = math.ceil(max_len / out_steps) * out_steps
    return np.stack(
        [pad_to_length(x, target_len, pad_value=pad_value) for x in inputs])
