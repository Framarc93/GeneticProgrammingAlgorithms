# Genetic Programming Algorithms

This folder contains the Genetic Programming (GP) algorithms developed by Francesco Marchetti. 
The algorithms are built using the DEAP library (https://github.com/deap/deap) and many functions
are a modification of the functions implemented in DEAP.

The folder contains the code to apply the following algorithms:
* Inclusive Genetic Programming (IGP) [1]
* Full Inclusive Genetic Programming (FIGP) [2]
* Python implementation of the Multi-Gene Genetic Programming (pyMGGP)

The IGP was developed by Francesco Marchetti during his PhD at the University of Strathclyde and originally published 
online at https://github.com/strath-ace/smart-ml/tree/master/GP/GP_Algorithms/IGP. It has been republished in this 
folder in order to keep it up to date. The IGP was used in 

The pyMGGP is a Python implementation of the MGGP originally introduced by Hinchliffe et al. [3] and later made 
available by Searson et al. [4] in the open source GPTIPS matlab library [5].

The source code to use the aforementioned algorithms is contained in the `src` folder, while some regression and control
applications examples are contained in the `examples` folder.



## References

1. Marchetti, F., Minisci, E. (2021). Inclusive Genetic Programming. In: Hu, T., Lourenço, N., Medvet, E. (eds) Genetic Programming. EuroGP 2021. Lecture Notes in Computer Science(), vol 12691. Springer, Cham. https://doi.org/10.1007/978-3-030-72812-0_4 
2. F. Marchetti, M. Castelli, I. Bakurov and L. Vanneschi, "Full Inclusive Genetic Programming," 2024 IEEE Congress on Evolutionary Computation (CEC), Yokohama, Japan, 2024, pp. 1-8, doi: 10.1109/CEC60901.2024.10611808.
3. Hinchliffe MP, Willis MJ, Hiden H, Tham MT, McKay B & Barton, GW. Modelling chemical process systems using a multi-gene genetic programming algorithm. In Genetic Programming: Proceedings of the First Annual Conference (late breaking papers), 56-65. The MIT Press,
USA, 1996.
4. Searson, Dominic P., David E. Leahy, and Mark J. Willis. "GPTIPS: an open source genetic programming toolbox for multigene symbolic regression." Proceedings of the International multiconference of engineers and computer scientists. Vol. 1. Citeseer, 2010.
5. https://www.mathworks.com/matlabcentral/fileexchange/68527-gptips-free-open-source-genetic-programming-and-symbolic-data-mining-matlab-toolbox



## Development and checks

Use Python 3.10–3.13 with the pinned scientific dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check src examples tests
ruff format --check src examples tests
pytest --cov=genetic_programming_algorithms --cov=examples --cov-report=term-missing
```

GitHub Actions runs these checks when a pull request is opened, updated with
new commits, or reopened (not on standalone pushes), for Python
3.10, 3.11, 3.12 and 3.13. Ruff enforces import ordering, unused-name checks,
modern Python syntax and consistent formatting. Existing public function names
are retained for compatibility. See https://docs.astral.sh/ruff/configuration/.

Tests use small seeded populations and synthetic data; dataset downloads are
mocked. Full benchmark campaigns are not run in CI. Coverage includes the
example scripts so untested script paths remain visible.

Regression dataset scaling is fitted on the training split only. Validation
and test values may therefore fall outside [0, 1]. Protected numerical
primitives no longer mutate input data. Impossible bloat limits raise
ValueError instead of hanging.
