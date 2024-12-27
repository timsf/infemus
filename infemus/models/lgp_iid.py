from typing import Iterator

import numpy as np
import numpy.typing as npt


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float64]


def sample_posterior(y: FloatArr, mean: FloatArr, cov: FloatArr, phi: float, ome: np.random.Generator
                     ) -> Iterator[FloatArr]:

    cmean, ccov = eval_post_moments(y, mean, cov, phi)
    cf_ccov = np.linalg.cholesky(ccov)
    while True:
        yield cmean + cf_ccov @ ome.standard_normal(len(y))


def eval_post_moments(y: FloatArr, mean: FloatArr, cov: FloatArr, phi: float) -> tuple[FloatArr, FloatArr]:

    ext_mean = np.hstack([mean, mean])
    ext_cov = np.vstack([np.hstack([cov + np.diag(np.repeat(phi ** 2, len(y))), cov]),
                         np.hstack([cov, cov])])
    cmean, ccov = update_joint(y, np.repeat([True, False], len(y)), ext_mean, ext_cov)
    return cmean, ccov


def regress(where: FloatArr, cov: FloatArr) -> tuple[FloatArr, FloatArr]:

    ccoefs = np.linalg.solve(cov[where][:, where], cov[where][:, ~where]).T
    ccov = cov[~where][:, ~where] - ccoefs @ cov[where][:, ~where]
    return ccoefs, ccov


def update_joint(y: FloatArr, where: FloatArr, mean: FloatArr, cov: FloatArr
                 ) -> tuple[FloatArr, FloatArr]:

    ccoefs, ccov = regress(where, cov)
    cmean = mean[~where] + ccoefs @ (y - mean[where])
    return cmean, ccov
