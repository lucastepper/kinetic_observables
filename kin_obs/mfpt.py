import numpy as np
from numba import njit


def find_barrier_minima(values, fe, barrier_region):
    """ Find the position of the barrier which is the bin with the highest free
    energy in barrier region (iterable: (min, max)). Then find the minimum left and right to it. """
    # barrier
    mask_barrier_region = np.logical_and(
        values >= barrier_region[0], values <= barrier_region[1]
    )
    value_barrier = values[mask_barrier_region][np.argmax(fe[mask_barrier_region])]
    fe_barrier = fe[mask_barrier_region][np.argmax(fe[mask_barrier_region])]
    # minimum left to barrier
    mask_left = values < value_barrier
    value_min_left = values[mask_left][np.argmin(fe[mask_left])]
    fe_min_left = fe[mask_left][np.argmin(fe[mask_left])]
    # minimum right of barrier
    mask_right = values > value_barrier
    value_min_right = values[mask_right][np.argmin(fe[mask_right])]
    fe_min_right = fe[mask_right][np.argmin(fe[mask_right])]
    return np.array(
        [
            [value_min_left, fe_min_left],
            [value_barrier, fe_barrier],
            [value_min_right, fe_min_right],
        ]
    )


@njit
def is_transition(x1, x2, barrier):
    """ I AM DOCSTRING. """
    if x1 <= barrier and x2 > barrier:
        return True
    if x1 > barrier and x2 <= barrier:
        return True
    return False


@njit
def get_ffpt_single(traj, start, end):
    """ I AM DOCSTRING. """
    count = 0
    ffpt_sum = 0.0
    start_found = -1
    for i in range(len(traj) - 1):
        if start_found < 0:
            if is_transition(traj[i], traj[i + 1], start):
                start_found = i
        elif is_transition(traj[i], traj[i + 1], end):
            ffpt_sum += i - start_found
            count += 1
            start_found = -1
    return ffpt_sum, count


@njit
def get_fpt_single(traj, start, end):
    """ I AM DOCSTRING. """
    count = 0
    fpt_sum = 0.0
    end_found = -1
    # iterate through from end, add all diffs from end found
    # to every start found, update end when found
    for i in range(1, len(traj)):
        i = len(traj) - i - 1
        if is_transition(traj[i], traj[i + 1], end):
            end_found = i
        if end_found >= 0 and is_transition(traj[i], traj[i + 1], start):
            fpt_sum += end_found - i
            count += 1
    return fpt_sum, count


def get_mfpt(trajs, edges_start, edges_end, dt, return_sum=False):
    """ I AM DOCSTRING. """
    if isinstance(trajs, np.ndarray) and trajs.ndim > 1:
        pass
    elif not isinstance(trajs, (list, tuple)):
        trajs = [trajs]
    fpts_sum = np.zeros((len(edges_start), len(edges_end)))
    counter = np.zeros((len(edges_start), len(edges_end)), dtype=np.int64)

    for traj in trajs:
        if isinstance(traj, str):
            traj = np.load(traj)
        for i, start in enumerate(edges_start):
            for j, end in enumerate(edges_end):
                fpt_sum, count = get_fpt_single(traj, start, end)
                fpts_sum[i, j] += fpt_sum
                counter[i, j] += count
    if return_sum:
        return fpts_sum, counter
    mfpts = np.zeros_like(fpts_sum)
    mfpts[counter != 0] = fpts_sum[counter != 0] * dt / counter[counter != 0]
    return mfpts


def get_mffpt(trajs, edges_start, edges_end, dt, return_sum=False):
    """ I AM DOCSTRING. """
    if isinstance(trajs, np.ndarray) and trajs.ndim > 1:
        pass
    elif not isinstance(trajs, (list, tuple)):
        trajs = [trajs]
    ffpts_sum = np.zeros((len(edges_start), len(edges_end)))
    counter = np.zeros((len(edges_start), len(edges_end)), dtype=np.int64)

    for traj in trajs:
        if isinstance(traj, str):
            traj = np.load(traj)
        for i, start in enumerate(edges_start):
            for j, end in enumerate(edges_end):
                ffpt_sum, count = get_ffpt_single(traj, start, end)
                ffpts_sum[i, j] += ffpt_sum
                counter[i, j] += count
    if return_sum:
        return ffpts_sum, counter
    mffpts = np.zeros_like(ffpts_sum)
    mffpts[counter != 0] = ffpts_sum[counter != 0] * dt / counter[counter != 0]
    return mffpts
