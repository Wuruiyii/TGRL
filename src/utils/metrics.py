import numpy as np


def normalised_makespan(cmax, cmax_ref):
    return cmax / cmax_ref if cmax_ref > 0 else float("inf")


def mean_tardiness(completion_times, due_dates, weights=None):
    tardiness = [max(0, c - d) for c, d in zip(completion_times, due_dates)]
    if weights:
        tardiness = [t * w for t, w in zip(tardiness, weights)]
    return np.mean(tardiness)


def machine_utilisation(busy_times, total_time, n_machines):
    return sum(busy_times) / (total_time * n_machines) if total_time > 0 else 0.0


def agv_utilisation(transport_times, total_time, n_agvs):
    return sum(transport_times) / (total_time * n_agvs) if total_time > 0 else 0.0


def wilcoxon_test(x, y):
    from scipy.stats import wilcoxon
    stat, p = wilcoxon(x, y, alternative="less")
    return stat, p


def cliffs_delta(x, y):
    n_x, n_y = len(x), len(y)
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    return (more - less) / (n_x * n_y)
