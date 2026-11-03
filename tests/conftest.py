import random

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def deterministic_randomness():
    random.seed(42)
    np.random.seed(42)
