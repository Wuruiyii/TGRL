"""Disruption profile parameter tables (Table 2, Table 4)."""

DISRUPTION_TYPE_PARAMS = {
    "machine_breakdown": {
        "arrival": "Poisson",
        "duration": "Exponential(1/mu_b)",
        "lambda_range": (0.01, 0.05),
        "mu_b": 0.1,
    },
    "urgent_insertion": {
        "arrival": "Poisson",
        "n_ops": "Uniform(2,6)",
        "lambda_range": (0.005, 0.02),
    },
    "processing_variation": {
        "per_operation": True,
        "sigma_range": (0.05, 0.20),
    },
    "agv_failure": {
        "arrival": "Poisson",
        "duration": "Exponential(1/mu_v)",
        "lambda_range": (0.002, 0.01),
        "mu_v": 0.2,
    },
    "agv_congestion": {
        "event_driven": True,
        "delay_coeff": 2.0,
    },
}
