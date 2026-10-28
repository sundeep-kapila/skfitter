"""Small helpers shared by the tests."""
import numpy as np


def random_params(fitter, seed=0):
    """Every free parameter of the model at a random value (small overlaps so S stays positive definite)."""
    rng = np.random.default_rng(seed)
    vec = []
    for k in fitter._get_param_keys():
        if k.startswith("s_"):
            vec.append(rng.uniform(-0.03, 0.03))
        elif k.startswith("eta"):
            vec.append(rng.uniform(1.5, 3.5))
        else:
            vec.append(rng.uniform(-1.0, 1.0))
    return fitter._param_vec_to_dict(np.array(vec))
