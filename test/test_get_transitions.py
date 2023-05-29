import os
import numpy as np
import matplotlib.pyplot as plt
import pytest
import kin_obs


np.random.seed(42)
DATA = {
    "test_data_1": {
        "traj": [-1.0, 0.0, 1.0, 2.0],
        "start": -1.5,
        "end": 0.5,
        "nbins": 2,
        "ref_transition_times": [1],
        "ref_transition_idxs": [0],
    },
    "test_data_2": {
        "traj": [-1.6, -1.4, -0.9, -0.8, -0.4, 0.6],
        "start": -1.5,
        "end": 1.0,
        # dx = 0.5
        "nbins": 5,
        "ref_transition_times": [1, 2, 4, 5, 5],
        "ref_transition_idxs": [0, 1, 2, 3, 4],
    },
    "test_data_3": {
        "traj": [0.6, 0.55, -0.6],
        "start": -1.5,
        "end": 1.0,
        # dx = 0.5
        "nbins": 5,
        "ref_transition_times": [2, 2, 2],
        "ref_transition_idxs": [4, 3, 2],
    },
}


@pytest.mark.parametrize("key", DATA.keys())
def test_get_transitions(key):
    """Get data from data dict and test mfpt function"""
    transition_times, transition_idxs = kin_obs.get_transition_times_idxs(
        np.array(DATA[key]["traj"]), DATA[key]["start"], DATA[key]["end"], DATA[key]["nbins"]
    )
    np.allclose(transition_times, DATA[key]["ref_transition_times"])
    np.allclose(transition_idxs, DATA[key]["ref_transition_idxs"])


def get_idx(x, start, end, dx, nbins):
    """This is a copy of the function in src/lib.rs"""
    if x < start:
        return 0
    if x > end:
        return nbins
    return int((x + 1.5) / dx) + 1


def main():
    """Test MFPT implementation with very easy data. If PLOT=true in env, plot."""

    key = "test_data_3"
    # values = np.linspace(-1.6, 1.5, 100)
    # dx = (DATA[key]["end"] - DATA[key]["start"]) / DATA[key]["nbins"]
    # idxs = [
    #     get_idx(x, DATA[key]["start"], DATA[key]["end"], dx, DATA[key]["nbins"]) for x in values
    # ]
    # plt.scatter(values, idxs)
    # plt.show()
    transition_times, transition_idxs = kin_obs.get_transition_times_idxs(
        np.array(DATA[key]["traj"]), DATA[key]["start"], DATA[key]["end"], DATA[key]["nbins"]
    )
    print(f"{transition_times=}", "ref: ", DATA[key]["ref_transition_times"])
    print(f"{transition_idxs=}", "ref: ", DATA[key]["ref_transition_idxs"])


if __name__ == "__main__":
    main()
