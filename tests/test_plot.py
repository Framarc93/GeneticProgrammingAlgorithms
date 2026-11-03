import matplotlib.pyplot as plt
import numpy as np

from examples.control.plot_functions import plot


def test_plot(tmp_path, monkeypatch):
    path = str(tmp_path) + "/"
    for name, value in {
        "x_ref": np.zeros((4, 4)),
        "u_ref": np.zeros(4),
        "time_points": np.arange(4),
        "best_fitness": 2.0,
        "computational_time": 1.0,
        "fitness_evol": np.ones((3, 2)),
        "x": np.zeros((4, 4)),
        "u": np.zeros(4),
    }.items():
        np.save(tmp_path / (name + ".npy"), value)
    (tmp_path / "best_ind_structure.txt").write_text("0")
    monkeypatch.setattr(plt, "show", lambda: None)
    plot(path, path, "IGP", 1, 3, 4)
    assert len(list((tmp_path / "Plots").glob("*.pdf"))) == 7
    plt.close("all")
