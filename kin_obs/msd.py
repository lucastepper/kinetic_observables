import numpy as np
from correlation import correlation


def msd(traj, trunc=None):
    """Compute the msd of a trajectory.
        Delta x^2(t) = sum_{dims} < (x(0) - x(t))^2 >
    Arguments:
        traj (np.ndarray (steps, dims)): Trajectory
        trunc (int): Truncate the msd at this many steps
    Returns:
        msd (np.ndarray (steps)): MSD
    """
    traj = np.asarray(traj)
    if trunc is None:
        trunc = len(traj)
    if traj.ndim == 1:
        traj = traj.reshape((-1, 1))
    output = np.zeros(trunc, dtype=float)
    sum_xx_corr = np.sum(
        [correlation(traj[:, i], traj[:, i], trunc=trunc) for i in range(traj.shape[1])], axis=0
    )
    # this is how one would compute msd(0), which should equate to something close machine precision
    # we leave it at 0, per the analytical prediction
    # output[0] = 2. * np.sum(traj ** 2) - 2. * sum_xx_corr[0] * len(traj)
    # compute the msd of the rest of the trajectory, up to trunc
    # we need the squared traj to trunc - 1 from the start and end of the trajectory
    # we write down each array seperately, assuming that trunc < len(traj) / 2
    # it still works for bigger truncs, just not as efficiently
    sum_traj_sqrd_start = np.sum(traj[: trunc - 1] ** 2, axis=1)
    sum_traj_sqrd_end = np.sum(traj[-(trunc - 1) :] ** 2, axis=1)
    # we need the sum over the entire traj ** 2, which we construct from the arrays and
    # part of the trajectory we have left out
    # this only works for trunc < len(traj) / 2
    if trunc < len(traj) / 2:
        sum_traj_sqrd = (
            np.sum(sum_traj_sqrd_start)
            + np.sum(sum_traj_sqrd_end)
            + np.sum(traj[trunc - 1 : -(trunc - 1)] ** 2)
        )
    else:
        sum_traj_sqrd = np.sum(traj**2)
    # assemble result
    output[1:] = (
        2.0 * sum_traj_sqrd - np.cumsum(sum_traj_sqrd_start) - np.cumsum(sum_traj_sqrd_end[::-1])
    ) / (len(traj) - 1 - np.arange(trunc - 1))
    output[1:] -= 2 * sum_xx_corr[1:]
    return output
