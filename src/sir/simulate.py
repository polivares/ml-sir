"""SIR simulation wrapper using summer.

Provides a single entry point to simulate SIR trajectories given (beta, gamma)
and initial conditions. By default it returns I(t); optionally returns full
S, I, R outputs and the time grid for downstream baselines and plotting.
Used by classical fitting routines that need to re-simulate curves.
"""


from typing import Tuple, Union

import numpy as np
from summer import CompartmentalModel


def simulate_sir(
    beta: float,
    gamma: float,
    s0: float = 90.0,
    i0: float = 10.0,
    r0: float = 0.0,
    t0: float = 0.0,
    t1: float = 100.0,
    dt: float = 0.1,
    return_full: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Simulate SIR with summer.

    If return_full is False, returns the infected time series I(t).
    If return_full is True, returns (times, outputs) where outputs has shape (T, 3).
    """
    # Basic input validation to avoid silent misconfiguration.
    if dt <= 0:
        raise ValueError("dt must be positive")
    if t1 <= t0:
        raise ValueError("t1 must be greater than t0")

    # Build a simple SIR system in summer.
    model = CompartmentalModel(
        times=[t0, t1],
        compartments=["S", "I", "R"],
        infectious_compartments=["I"],
        timestep=dt,
    )
    model.set_initial_population(distribution={"S": s0, "I": i0, "R": r0})
    # beta controls transmission; gamma controls recovery.
    model.add_infection_frequency_flow(
        name="infection", contact_rate=beta, source="S", dest="I"
    )
    model.add_transition_flow(
        name="recovery", fractional_rate=gamma, source="I", dest="R"
    )
    model.run()

    # Extract outputs for downstream use.
    outputs = model.outputs
    times = model.times
    if return_full:
        return times, outputs
    # Return only I(t) by default.
    return outputs[:, 1]
