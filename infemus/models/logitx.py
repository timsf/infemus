from typing import Callable, Iterator

import numpy as np
import numpy.typing as npt

from infemus.tools.metropolis_uv import LatentGaussSampler
from scipy.special import expit


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float64]
ParamSpace = tuple[list[FloatArr], FloatArr]
DispParamSpace = tuple[list[FloatArr], FloatArr, float]
PartFunc = Callable[[FloatArr], tuple[FloatArr, FloatArr, FloatArr]]


def sample_posterior(
    y: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    tau0: float,
    tau: FloatArr,
    collapse: bool,
    ome: np.random.Generator,
) -> Iterator[tuple[float, FloatArr]]:

    alp0 = 0
    alp = [np.zeros(j_) for j_ in j]

    i_ord = np.argsort(i, 0)
    samplers = [LatentGaussSampler(j_) for j_ in j]

    while True:
        yield update_coefs(y, n, j, i, i_ord, alp0, alp, 0, tau0, tau, 1, collapse, eval_part, samplers, ome)


def eval_part(eta: FloatArr) -> tuple[FloatArr, FloatArr, FloatArr]:

    mu = expit(eta)
    return np.logaddexp(0, eta), mu, mu * (1 - mu)


def update_coefs(
    y1: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    i_ord: IntArr,
    alp0: float,
    alp: list[FloatArr],
    eps: float,
    tau0: float,
    tau: FloatArr,
    phi: float,
    collapse: bool,
    eval_part: PartFunc,
    samplers: list[LatentGaussSampler],
    ome: np.random.Generator,
) -> tuple[float, FloatArr]:

    new_alp0, new_alp = alp0, alp.copy()
    for k_, (tau_, sampler_) in enumerate(zip(tau, samplers)):
        new_alp0, new_alp[k_] = update_single_coef(y1, n, j, i, i_ord, k_, new_alp0, new_alp, eps, tau0, tau_,
                                                   phi, collapse, eval_part, sampler_, ome)
    return new_alp0, new_alp


def update_single_coef(
    y1: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    i_ord: IntArr,
    k_: int,
    alp0: float,
    alp: list[FloatArr],
    eps: float,
    tau0: float,
    tau_: float,
    phi: float,
    collapse: bool,
    eval_part: PartFunc,
    sampler: LatentGaussSampler,
    ome: np.random.Generator,
) -> tuple[float, FloatArr]:

    def eval_log_p(b: FloatArr) -> tuple[FloatArr, FloatArr, FloatArr]:
        log_p, dk_log_p, d2k_log_p = eval_kernel(y1, n, j, i, i_ord, alp0 + eps, alp[:k_] + [b - alp0] + alp[(k_ + 1):],
                                                 eval_part, k_)
        return log_p / phi, dk_log_p / phi, d2k_log_p / phi

    new_bet_ = sampler.sample(alp[k_] + alp0, np.repeat(alp0, len(alp[k_])), np.repeat(tau_, len(alp[k_])),
                              eval_log_p, ome)
    if collapse:
        new_alp0 = ome.normal(tau_ * np.sum(new_bet_) / (tau0 + tau_ * len(alp[k_])), 1 / np.sqrt(tau0 + tau_ * len(alp[k_])))
        new_alp_ = new_bet_ - new_alp0
    else:
        new_alp_ = new_bet_ - alp0
        new_alp0 = update_intercept(y1, n, j, i, i_ord, alp0, alp[:k_] + [new_alp_] + alp[(k_ + 1):], eps,
                                    0, phi, eval_part, ome)
    return new_alp0, new_alp_


def update_intercept(
    y1: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    i_ord: IntArr,
    alp0: float,
    alp: list[FloatArr],
    eps: float,
    tau0: float,
    phi: float,
    eval_part: PartFunc,
    ome: np.random.Generator,
) -> float:

    def eval_log_p(a: FloatArr) -> tuple[FloatArr, FloatArr, FloatArr]:
        log_p, dk_log_p, d2k_log_p = eval_kernel(y1, n, j, i, i_ord, a[0] + eps, alp, eval_part, None)
        return log_p / phi, dk_log_p / phi, d2k_log_p / phi

    sampler = LatentGaussSampler(1)
    return sampler.sample(np.float64([alp0]), np.zeros(1), np.float64([tau0 if tau0 != 0 else np.finfo(float).eps]),
                          eval_log_p, ome)[0]


def eval_kernel(
    y1: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    i_ord: IntArr,
    alp0: float,
    alp: list[FloatArr],
    eval_part: PartFunc,
    k_: int | None = None,
) -> tuple[FloatArr, FloatArr, FloatArr]:

    eta = eval_lin_pred(i, alp0, alp)
    log_p, d_log_p, d2_log_p = eval_densities(y1, n, eta, eval_part)

    if k_ is not None:
        brk = np.cumsum(np.bincount(i[:, k_], minlength=j[k_]))[:-1]
        return groupby(log_p, i_ord[:, k_], brk), groupby(d_log_p, i_ord[:, k_], brk), \
               groupby(d2_log_p, i_ord[:, k_], brk)
    return np.sum(log_p, 0)[np.newaxis], np.sum(d_log_p, 0)[np.newaxis], np.sum(d2_log_p, 0)[np.newaxis]


def eval_densities(
    y1: FloatArr, 
    n: FloatArr, 
    eta: FloatArr, 
    eval_part: PartFunc
) -> tuple[FloatArr, FloatArr, FloatArr]:

    part, d_part, d2_part = eval_part(eta)
    log_f = y1 * eta - n * part
    d_log_f = y1 - n * d_part
    d2_log_f = - n * d2_part
    return log_f, d_log_f, d2_log_f


def groupby(arr: FloatArr, ord: FloatArr, brk: FloatArr) -> FloatArr:

    return np.float64([np.sum(a) for a in np.split(arr[ord], brk)])


def eval_lin_pred(i: IntArr, alp0: float, alp: list[FloatArr]) -> FloatArr:

    return alp0 + np.sum(np.float64([alp_[i_] for alp_, i_ in zip(alp, i.T)]), 0)
