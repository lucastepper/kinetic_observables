import numpy as np
import pytest
from numba import njit
import kin_obs

np.random.seed(45)
PLOT = False


@njit
def calc_msd_simple(x, trunc=None):
    """Compute the msd of a trajectory by direct summation.
        Delta x^2(t) = sum_{dims} < (x(0) - x(t))^2 >
    Arguments:
        traj (np.ndarray (steps, dims)): Trajectory
        trunc (int): Truncate the msd at this many steps
    Returns:
        msd (np.ndarray (steps)): MSD
    """
    if trunc is None:
        trunc = len(x)
    n = len(x)
    msd = np.zeros(trunc)
    for s in range(1, trunc):
        x2 = 0.0
        for i in range(n - s):
            x2 += (x[i + s] - x[i]) ** 2
        x2 /= n - s
        msd[s] = x2
    return msd


@pytest.mark.parametrize("ndim", [1, 2])
def test_msd(ndim):
    """Test the msd vs a reference implementation."""
    trunc = 800
    test_traj = np.cumsum(np.random.normal(size=(int(1e3), ndim), scale=0.01), axis=0)
    # when computing the msd for multiple dimensions, we need to sum over the dimensions
    msd_ref = np.sum([calc_msd_simple(test_traj[:, i], trunc) for i in range(ndim)], axis=1)
    # pylint: disable=too-many-function-args
    msd_test = kin_obs.msd(test_traj, trunc)
    if PLOT:
        import matplotlib.pyplot as plt

        plt.plot(msd_ref)
        plt.plot(msd_test)
        plt.show()


if __name__ == "__main__":
    test_msd(2)
