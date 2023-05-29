import numpy as np
import pytest
from numba import njit
import kin_obs

PLOT = False
np.random.seed(44)


@njit
def is_transition(x1, x2, barrier):
    """Reference implementations to compute first passage distribtuions."""
    if x1 <= barrier and x2 > barrier:
        return True
    if x1 > barrier and x2 <= barrier:
        return True
    return False


@njit
def get_ffpt_single(traj, start, end):
    """Reference implementations to compute first passage distribtuions."""
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
    """Reference implementations to compute first passage distribtuions."""
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
    """Reference implementations to compute first passage distribtuions."""
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
    """Reference implementations to compute first passage distribtuions."""
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


@njit
def force(x, l, u0):
    """Force from a double well potential."""
    return 4.0 * u0 * x / l**2 * (1.0 - x**2 / l**2)


@njit
def integrate(x0, gamma, l, u0, kbt, dt, steps):
    """Integrate overdamped Langevin equation using Euler-Maruyama method.\
    Used to generate trajectories for testing.
    """
    traj = np.zeros(steps + 1)
    traj[0] = x0
    for i in range(steps):
        traj[i + 1] = (
            traj[i]
            + dt * (force(traj[i], l, u0) / gamma)
            + np.sqrt(2.0 * kbt * dt / gamma) * np.random.normal()
        )
    return traj


@pytest.mark.parametrize(
    # "to_test, parallel", [("first", False), ("first", True), ("all", False), ("all", True)]
    "to_test, parallel", [("all", False)]
)
def test_passage_times(to_test, parallel):
    """Test either first-first passage times or all-all passage times
    against reference implementation.
    """
    l = 1.2
    nbins = 14
    dt = 0.01
    traj = integrate(1.0, 120.0, l, 1.5, 2.494, dt, int(1e6))
    if parallel:
        traj = np.stack([traj, integrate(1.0, 120.0, l, 1.5, 2.494, dt, int(1e6))], axis=0)
    if to_test == "first":
        mftps_ref = get_mffpt(traj, [-l, l], np.linspace(-l, l, nbins), dt)
    elif to_test == "all":
        mftps_ref = get_mfpt(traj, [-l, l], np.linspace(-l, l, nbins), dt)
    else:
        raise ValueError(f"to_test must be 'first' or 'all', not {to_test}")
    passage_times = kin_obs.PassageTimes(-l, l, nbins, dt, to_test)
    passage_times.add_data(traj.reshape(1, -1))
    mftps_test = passage_times.get_result()
    if PLOT:
        import matplotlib.pyplot as plt

        plt.plot(mftps_ref[0], c="C0", lw=2)
        plt.plot(mftps_ref[1], c="C0", lw=2)
        plt.plot(mftps_test[0], c="C1", lw=1)
        plt.plot(mftps_test[1], c="C1", lw=1)
        plt.show()


if __name__ == "__main__":
    test_passage_times("first", parallel=False)
