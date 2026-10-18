"""Feature extraction utilities for operation, machine and AGV nodes.

Feature dimensions:
    d_O = 8  (4 status one-hot + mean_p + var_p + remaining + pos_norm + slack)  -- actually 9, padded
    d_M = 7  (3 status one-hot + util + qlen + eta + bd_hist)
    d_V = 7  (2 pos + 4 status one-hot + battery + cong)  -- actually 8
"""
import numpy as np


def normalise_processing_time(p, p_max=50.0):
    return p / p_max


def compute_due_date_slack(due_date, current_time):
    if due_date <= 0:
        return 0.0
    return (due_date - current_time) / due_date


def sliding_window_utilisation(history, window=20):
    if len(history) < window:
        return np.mean(history) if history else 0.0
    return np.mean(history[-window:])


def normalise_position(pos, grid_dims=(5, 4)):
    return (pos[0] / grid_dims[0], pos[1] / grid_dims[1])
