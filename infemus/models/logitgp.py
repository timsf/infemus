from typing import Iterator

import numpy as np
import numpy.typing as npt
from scipy.special import expit

from infemus.tools.metropolis_mv import LatentGaussSampler


BoolArr = npt.NDArray[np.bool_]
IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float64]


def sample_posterior(
    y: BoolArr, 
    mean: FloatArr, 
    eig_cov: tuple[FloatArr, FloatArr], 
    ome: np.random.Generator,
) -> Iterator[FloatArr]:

    sampler = LatentGaussSampler(1)
    z = ome.standard_normal(len(y))
    while True:
        z = update(y, mean, eig_cov, z, sampler, ome)
        yield z


def update(
    y: BoolArr, 
    mean: FloatArr, 
    eig_cov: tuple[FloatArr, FloatArr], 
    z_nil: FloatArr, 
    sampler: LatentGaussSampler, 
    ome: np.random.Generator,
) -> FloatArr:
    
    return sampler.sample(z_nil[np.newaxis], mean, eig_cov[1], 1 / eig_cov[0], lambda z_: eval_loglik(np.int_(y), np.ones(len(y)), z_[0]), ome)[0]
    

def eval_loglik(y1: IntArr, n: IntArr, z: FloatArr) -> tuple[float, FloatArr]:

    mu = expit(z)
    part, d_part = np.logaddexp(0, z), mu
    log_p = y1 * z - n * part
    d_log_p = y1 - n * d_part
    return np.sum(log_p)[np.newaxis], d_log_p[np.newaxis]
