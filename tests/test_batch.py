import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from examples.regression.main_regression import run_experiments
from examples.regression.result_analysis import collect_results


def test_collect_results_statistics_and_short_runs(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    plotted = []
    original_scatter = Axes.scatter

    def capture_scatter(self, x, y, *args, **kwargs):
        plotted.append((np.asarray(x), np.asarray(y)))
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", capture_scatter)
    (tmp_path / "batch.json").write_text(json.dumps({"runs": ["Sim0", "Sim1"]}))
    for i, history in enumerate(([4.0, 2.0], [2.0])):
        folder = tmp_path / f"Sim{i}"
        folder.mkdir()
        record = {
            "experiment": i,
            "seed": i,
            "selected_generation": 0,
            "train_rmse": 1.0 + i,
            "validation_rmse": 2.0 + i,
            "test_rmse": 3.0 + i,
            "nodes": 3 + i,
            "evolution_seconds": 1.0 + i,
            "training_history": history,
        }
        (folder / "result.json").write_text(json.dumps(record))
    table, evolution = collect_results(tmp_path)
    np.testing.assert_allclose(evolution["mean"], [3, 2])
    np.testing.assert_allclose(evolution["std"], [np.sqrt(2), 0])
    assert evolution["active_runs"].tolist() == [2, 1]
    assert table["validation_rmse"].tolist() == [2, 3]
    assert (tmp_path / "training_evolution.png").stat().st_size > 0
    assert (tmp_path / "best_fitness_boxplots.png").stat().st_size > 0
    assert (tmp_path / "complexity_vs_fitness.png").stat().st_size > 0
    assert (tmp_path / "runtime_vs_fitness.png").stat().st_size > 0
    np.testing.assert_array_equal(plotted[-2][0], [3, 4])
    np.testing.assert_array_equal(plotted[-2][1], [3, 4])
    np.testing.assert_array_equal(plotted[-1][0], [1, 2])
    np.testing.assert_array_equal(plotted[-1][1], [3, 4])


@pytest.mark.parametrize("algorithm", ["IGP", "FIGP", "MGGP"])
def test_small_batch(algorithm, tmp_path):
    config = yaml.safe_load(Path("examples/regression/regression_config.yaml").read_text())
    config.update(ntot=2, size_pop=8, size_gen=2, init_repeat=1, save_pop=False, fit_tol=None)
    output = run_experiments(
        config, algorithm=algorithm, benchmark="synthetic", output=tmp_path / algorithm, seed=7
    )
    manifest = json.loads((output / "batch.json").read_text())
    assert manifest["runs"] == ["Sim0", "Sim1"]
    for i, run in enumerate(manifest["runs"]):
        result = json.loads((output / run / "result.json").read_text())
        candidates = json.loads((output / run / "generation_scores.json").read_text())
        best = min(candidates, key=lambda c: c["test_rmse"])
        assert result["seed"] == 7 + i
        assert len(result["training_history"]) == 2
        for key in ("train_rmse", "validation_rmse", "test_rmse", "expression"):
            assert result[key] == best[key]
    with pytest.raises(FileExistsError):
        run_experiments(config, benchmark="synthetic", output=output)


def test_invalid_experiment_count(tmp_path):
    with pytest.raises(ValueError):
        run_experiments({"ntot": 0}, output=tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
