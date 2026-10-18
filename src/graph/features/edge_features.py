"""Edge feature computation for the six edge types."""


def eligibility_edge_feature(proc_time, p_max=50.0):
    return proc_time / p_max


def assignment_edge_features(start_time, completion_time, cmax_est):
    if cmax_est <= 0:
        return (0.0, 0.0)
    return (start_time / cmax_est, completion_time / cmax_est)


def transport_edge_feature(transport_time, tau_max=30.0):
    return transport_time / tau_max


def guidepath_adj_feature(distance, max_dist=10.0):
    return distance / max_dist


def congestion_edge_feature(delay, max_delay=20.0):
    return min(delay / max_delay, 1.0)
