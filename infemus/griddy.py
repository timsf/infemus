from typing import Callable, Iterator, TypeVar

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float64]
Hyper = TypeVar('H')
Params = TypeVar('P')
Data = TypeVar('D')


def est_mlik(
    lams: list[Hyper], 
    data: Data,
    instantiate_sampler: Callable[[Data, Hyper], Iterator[Params]],
    eval_logprior: Callable[[Params, Hyper], float],
    rng: np.random.Generator, 
    n_samples: int, 
    n_burnin: int,
) -> FloatArr:

    sampler = sample_marginal(lams, data, instantiate_sampler, eval_logprior, rng)
    for _, idx_ in zip(range(n_burnin * len(lams)), sampler):
        continue
    idx = [idx_]
    for _, idx_ in zip(range(n_samples * len(lams)), sampler):
        idx.append(idx_)
    counts = np.bincount(idx, minlength=len(lams))
    return counts / np.sum(counts)


def sample_marginal(
    lams: list[Hyper],
    data: Data,
    instantiate_sampler: Callable[[Data, Hyper], Iterator[Params]],
    eval_logprior: Callable[[Params, Hyper], float],
    rng: np.random.Generator,
) -> Iterator[int]:

    idx = rng.choice(len(lams))
    while True:
        the = next(instantiate_sampler(data, lams[idx], rng))
        log_psi = np.array([eval_logprior(data, the, lams_) for lams_ in lams])
        idx = rng.choice(len(lams), p=np.exp(log_psi - logsumexp(log_psi)))
        yield idx
