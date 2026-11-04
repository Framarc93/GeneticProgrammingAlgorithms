# WORK IN PROGRESS
## Repeated experiments and analysis

Set `ntot` in `regression_config.yaml`, or override it on the command line:

```sh
python -m examples.regression.main_regression --experiments 2 --population 20 --generations 6 --init-repeat 2 --benchmark synthetic --seed 42 --cpus 1
```

The default benchmark is `503_wind`; `synthetic` uses 120 samples from
`y = x0**2 + x1` without downloading data. `--algorithm` accepts IGP, FIGP,
and MGGP. `--output` selects a new batch directory; existing directories are
rejected to protect previous results. Without it, a timestamped directory is used.
Configuration values remain the defaults for population size, generations,
initialization repetitions and experiment count. Run with `--help` for options.

Each batch saves a shared, training-scaled data split, its configuration,
per-experiment seeds, generation champion checkpoints, all champion scores,
selected expressions, node counts, and evolution runtimes. Seeds increment from
`--seed`; the data split stays fixed across experiments. Generation zero is the
initial population. Champion checkpoints are always saved for post-evolution
selection, even if `save_gen` is false in the input configuration.

Selection preserves the example's existing rule: choose the generation training
champion with the smallest test RMSE. Training, validation, and test boxplot
values all refer to that same chosen individual within each experiment.
All RMSE values use the scaled target, not the original physical units.

`collect_results(batch_directory)` in `result_analysis.py` reads the completed
runs listed in `batch.json` and produces:

- `training_evolution.png`: mean generation-best training fitness across runs,
  with a ±1 sample-standard-deviation band and individual run traces.
- `best_fitness_boxplots.png`: training/validation/test scores of each selected
  champion, shown as boxplots.
- `complexity_vs_fitness.png`: total GP tree node count versus test RMSE of
  each selected champion (for MGGP, nodes summed across genes; fitted linear
  coefficients are not counted).
- `runtime_vs_fitness.png`: full evolution runtime versus selected champion
  test RMSE. Runtime includes population initialization and checkpoint saving,
  but excludes dataset preparation, post-evolution scoring and plotting. It is
  the whole run's runtime, not the time at which the selected champion appeared.
- `experiment_results.csv` and `training_evolution.csv`: numerical results.

Scatter points are labelled by run, starting at Run 1. Smaller models, shorter
runtimes and lower RMSE lie toward the lower left. These plots show associations
across runs; they do not establish that extra runtime or complexity causes a
change in fitness. Re-running `collect_results` regenerates plots from saved
results without re-running evolution.

Shorter, early-stopped runs carry their final training fitness forward to the
longest completed run. The CSV includes the number of still-active runs.
For one run the plotted spread is zero; two runs are a demonstration, not a
reliable characterization of variability.

Further analyses: success rate at a chosen error threshold, and variation over multiple data splits.
