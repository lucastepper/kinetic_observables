import os
import numpy as np
import matplotlib.pyplot as plt
import pytest
import kin_obs


np.random.seed(42)
DATA = {
    "test_data_1": {
        "traj": [
            -1.0,
            0.0,
            1.0,
            2.0,
        ],
        "dt": 0.02,
        "starts": [-0.5],
        "ends": [0.5],
        "ref_mfpt": [[0.02]],
        "ref_mffpt": [[0.02]],
    },
    "test_data_2": {
        "traj": [
            [
                -1.0,
                0.0,
                1.0,
                2.0,
            ]
        ],
        "dt": 0.02,
        "starts": [-0.5],
        "ends": [0.5],
        "ref_mfpt": [[0.02]],
        "ref_mffpt": [[0.02]],
    },
    "test_data_3": {
        "traj": np.cumsum(0.02 * np.random.randn(100)),
        "dt": 0.02,
        "starts": [-0.1],
        "ends": [-0.242],
        "ref_mfpt": [[0.6485714285714286]],
        "ref_mffpt": [[0.74]],
    },
}


def is_transition(x1, x2, barrier):
    """Check if two points are in different states."""
    if x1 <= barrier and x2 > barrier:
        return True
    if x1 > barrier and x2 <= barrier:
        return True
    False


def plot(key):
    """Plot trajectory, mark crossings"""
    plot_range = (DATA[key]["traj"].min(), DATA[key]["traj"].max())
    plot_range_len = abs(plot_range[1] - plot_range[0])
    plot_range = (plot_range[0] - plot_range_len / 20, plot_range[1] + plot_range_len / 10)
    time = np.arange(len(DATA[key]["traj"])) * DATA[key]["dt"]
    plt.plot(time, DATA[key]["traj"])
    plt.ylim(plot_range)
    # plot barrier crossings start
    counter = 1
    for i, start in enumerate(DATA[key]["starts"]):
        label = f"start_{i + 1}"
        for j, (x1, x2) in enumerate(zip(DATA[key]["traj"][:-1], DATA[key]["traj"][1:])):
            if is_transition(x1, x2, start):
                print(f"start_{i + 1} crossing {j + 1}")
                plt.plot(
                    100 * [(j + 0.5) * DATA[key]["dt"]],
                    np.linspace(*plot_range, 100),
                    linestyle=":",
                    label=label,
                    color=f"C{counter}",
                )
                label = None
        if not label:
            counter += 1
    # plot barrier crossings end
    for i, end in enumerate(DATA[key]["ends"]):
        label = f"end_{i + 1}"
        for j, (x1, x2) in enumerate(zip(DATA[key]["traj"][:-1], DATA[key]["traj"][1:])):
            if is_transition(x1, x2, end):
                print(f"end_{i + 1} crossing {j + 1}")
                plt.plot(
                    100 * [(j + 0.5) * DATA[key]["dt"]],
                    np.linspace(*plot_range, 100),
                    linestyle=":",
                    label=label,
                    color=f"C{counter}",
                )
                label = None
        if not label:
            counter += 1
    plt.legend()
    plt.show()


@pytest.mark.parametrize("key", DATA.keys())
def test_passage_time(key):
    """Get data from data dict and test mfpt function"""
    for method in ["first", "all"]:
        passage_times = kin_obs.PassageTimes(
            DATA[key]["starts"], DATA[key]["ends"], DATA[key]["dt"], method
        )
        passage_times.add_data(np.array(DATA[key]["traj"]))
        print(passage_times.__repr__())
        if method == "first":
            np.testing.assert_allclose(passage_times.get_result(), DATA[key]["ref_mffpt"])
        if method == "all":
            np.testing.assert_allclose(passage_times.get_result(), DATA[key]["ref_mfpt"])


def main():
    """Test MFPT implementation with very easy data. If PLOT=true in env, plot."""
    for key in DATA:
        test_passage_time(key)
    if os.getenv("PLOT", "false") == "true":
        plot("test_data_3")


if __name__ == "__main__":
    main()
