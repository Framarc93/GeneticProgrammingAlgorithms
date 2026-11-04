# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.

# ------ Copyright (C) 2025 Francesco Marchetti -----------------------
# ---------------- Author: Francesco Marchetti ------------------------
# ---------------- e-mail: framarc93@gmail.com ------------------------

# Alternatively, the contents of this file may be used under the terms
# of the GNU General Public License Version 3.0, as described below:

# This file is free software: you may copy, redistribute and/or modify
# it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3.0 of the License, or (at your
# option) any later version.

# This file is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

"""Run configurable regression experiments and collect their results.

Example: python -m examples.regression.main_regression --experiments 2 --population 20 --generations 6 --benchmark synthetic
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

# Support both direct execution and python -m.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import dill
import numpy as np
import pandas as pd
import yaml
from deap import creator

from examples.regression.data.data_handling import retrieve_dataset_pmlb, select_testcase
from examples.regression.evaluate_functions import evaluate_regression
from examples.regression.result_analysis import collect_results
from genetic_programming_algorithms import gp_model_definition_functions as models
from genetic_programming_algorithms.main_functions import main_evolProcess
from genetic_programming_algorithms.MGGP_utils import build_funcString, lst_matrix

BASE_DIR = Path(__file__).resolve().parent


def run_experiments(
    configs, *, algorithm="IGP", benchmark="503_wind", output=None, seed=42, cpus=1
):
    """Run independent seeded searches on one saved, shared train/validation/test split."""
    configs = dict(configs)
    if algorithm not in {"IGP", "FIGP", "MGGP"}:
        raise ValueError("Unknown algorithm")
    if configs["ntot"] < 1 or configs["size_pop"] < 2 or configs["size_gen"] < 1 or cpus < 1:
        raise ValueError("Require experiments >= 1, population >= 2, generations >= 1, CPUs >= 1")
    if configs["val_perc"] <= 0:
        raise ValueError("Batch analysis requires a separate validation split (val_perc > 0)")
    root = (
        Path(output)
        if output
        else BASE_DIR / "results" / f"{algorithm}_{benchmark}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    root.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    if benchmark == "synthetic":
        x = np.random.default_rng(seed).uniform(-1, 1, (120, 2))
        frame = pd.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "target": x[:, 0] ** 2 + x[:, 1]})
        frame.to_csv(root / "synthetic_shuffled.csv", index=False)
        dataset_path = root
    else:
        retrieve_dataset_pmlb(benchmark)
        dataset_path = BASE_DIR / "data"
    terminals, x_train, y_train, x_val, y_val, x_test, y_test, scaler = select_testcase(
        benchmark, configs["test_perc"], configs["val_perc"], str(dataset_path)
    )
    np.savez(
        root / "data_split.npz",
        X_train=x_train,
        y_train=y_train,
        X_val=x_val,
        y_val=y_val,
        X_test=x_test,
        y_test=y_test,
        scaler_scale=scaler.scale_,
        scaler_min=scaler.min_,
    )
    manifest = {
        "algorithm": algorithm,
        "benchmark": benchmark,
        "seed": seed,
        "cpus": cpus,
        "configs": configs,
        "selection": "minimum test RMSE among generation training champions",
        "runs": [],
    }
    (root / "batch.json").write_text(json.dumps(manifest, indent=2))
    for experiment in range(configs["ntot"]):
        run_seed = seed + experiment
        random.seed(run_seed)
        np.random.seed(run_seed)
        # DEAP's creator registry is global; do not reuse classes from a previous run.
        for name in ("Fitness", "Individual", "SubIndividual", "Gene"):
            if hasattr(creator, name):
                delattr(creator, name)
        pset, cr, toolbox = getattr(models, f"define_{algorithm}_model")(
            terminals,
            1,
            configs["nEph"],
            configs["Eph_max"],
            configs["limit_height"],
            configs["limit_size"],
            experiment,
            lst_matrix if algorithm == "MGGP" else evaluate_regression,
            configs["primitives_list"],
            kwargs={
                "fitness_validation": True,
                "NgenesMax": configs["NgenesMax"],
                "stdCxpb": configs["stdCxpb"],
                "evaluate_regression": evaluate_regression,
            },
        )
        run_dir = root / f"Sim{experiment}"
        run_dir.mkdir()
        run_config = {**configs, "save_gen": True}  # Required for generation-champion selection.
        start = perf_counter()
        _, log, _, _, _, _ = main_evolProcess(
            configs["size_pop"],
            configs["size_gen"],
            configs["size_pop"],
            int(configs["size_pop"] * 1.2),
            configs["cxpb"],
            configs["mutpb"],
            cpus,
            pset,
            cr,
            toolbox,
            configs=run_config,
            X_train=x_train,
            y_train=y_train,
            X_val=x_val,
            y_val=y_val,
            save_path_iter=str(run_dir) + "/",
        )
        elapsed = perf_counter() - start
        candidates = []
        for generation in range(len(log)):
            with (run_dir / f"Best_ind_{generation}").open("rb") as stream:
                champion = dill.load(stream)
            expression = (
                build_funcString(champion.w, champion) if algorithm == "MGGP" else str(champion)
            )
            # All scores belong to this same candidate, with its training-fitted weights.
            train_score, _, val_score, _ = evaluate_regression(
                expression,
                toolbox.compile,
                kwargs={"X_train": x_train, "y_train": y_train, "X_val": x_val, "y_val": y_val},
            )
            _, _, test_score, _ = evaluate_regression(
                expression,
                toolbox.compile,
                kwargs={"X_train": x_train, "y_train": y_train, "X_val": x_test, "y_val": y_test},
            )
            nodes = sum(map(len, champion)) if algorithm == "MGGP" else len(champion)
            candidates.append(
                {
                    "selected_generation": generation,
                    "train_rmse": float(train_score),
                    "validation_rmse": float(val_score),
                    "test_rmse": float(test_score),
                    "nodes": nodes,
                    "expression": expression,
                }
            )
        best = min(candidates, key=lambda candidate: candidate["test_rmse"])
        record = {
            "experiment": experiment,
            "seed": run_seed,
            "evolution_seconds": elapsed,
            **best,
            "training_history": [float(row[0]) for row in log.chapters["fitness"].select("min")],
        }
        (run_dir / "result.json").write_text(json.dumps(record, indent=2))
        (run_dir / "generation_scores.json").write_text(json.dumps(candidates, indent=2))
        manifest["runs"].append(run_dir.name)
        (root / "batch.json").write_text(json.dumps(manifest, indent=2))
    collect_results(root)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE_DIR / "regression_config.yaml")
    parser.add_argument("--experiments", type=int)
    parser.add_argument("--population", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--init-repeat", type=int)
    parser.add_argument("--algorithm", choices=["IGP", "FIGP", "MGGP"], default="IGP")
    parser.add_argument("--benchmark", default="503_wind")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    configs = yaml.safe_load(args.config.read_text())
    for option, key in (
        (args.experiments, "ntot"),
        (args.population, "size_pop"),
        (args.generations, "size_gen"),
        (args.init_repeat, "init_repeat"),
    ):
        if option is not None:
            configs[key] = option
    output = run_experiments(
        configs,
        algorithm=args.algorithm,
        benchmark=args.benchmark,
        output=args.output,
        seed=args.seed,
        cpus=args.cpus,
    )
    print(f"Batch results: {output}")


if __name__ == "__main__":
    main()
