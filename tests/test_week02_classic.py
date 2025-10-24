import hashlib
import sys
from collections.abc import Generator

sys.path.append('solutions')
sys.path.append("../")

import numpy as np
import pytest
from pytest import FixtureRequest
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

import templates.week02 as week02
from utils import (
    cache_data_stream_per_function,
    create_random_support,
    parameterize_classical_tests,
)

NUM_GAMES = 9
NUM_ZERO_SUM_GAMES = 2
BASE_SEED = 181


@pytest.fixture
def rng(request: FixtureRequest) -> np.random.Generator:
    """Create a random seed based on the function name to ensure reproducibility."""

    digest = hashlib.md5(request.node.originalname.encode()).hexdigest()
    seed = (BASE_SEED + int(digest, 16)) % (2**32)

    return np.random.default_rng(seed)


@pytest.fixture
@cache_data_stream_per_function
def general_sum_data_stream(request: FixtureRequest, rng: np.random.Generator) -> Generator:
    return parameterize_classical_tests(False, rng)


@pytest.fixture
@cache_data_stream_per_function
def zero_sum_data_stream(request: FixtureRequest, rng: np.random.Generator) -> Generator:
    return parameterize_classical_tests(True, rng)


@pytest.mark.parametrize('general_sum_data_stream', range(NUM_GAMES), indirect=True)
def test_verify_support(
    general_sum_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
    rng: np.random.Generator,
) -> None:
    row_matrix, col_matrix, *_ = next(general_sum_data_stream)
    row_support = create_random_support(row_matrix.shape[0], rng)
    col_support = create_random_support(col_matrix.shape[1], rng)

    col_probs = week02.verify_support(row_matrix, row_support, col_support)
    row_probs = week02.verify_support(np.transpose(col_matrix), col_support, row_support)

    # If no valid probabilities were found, use zeros for regression testing
    row_probs = row_probs if np.any(row_probs) else np.zeros_like(row_support, np.float64)
    col_probs = col_probs if np.any(col_probs) else np.zeros_like(col_support, np.float64)

    assert row_probs.dtype == np.float64, 'Incorrect dtype!'
    assert col_probs.dtype == np.float64, 'Incorrect dtype!'

    ndarrays_regression.check(
        {'row_probs': row_probs, 'col_probs': col_probs},
        f'{request.node.originalname}{request.node.callspec.indices["general_sum_data_stream"]}',
    )


@pytest.mark.parametrize('general_sum_data_stream', range(NUM_GAMES), indirect=True)
def test_support_enumeration(
    general_sum_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
) -> None:
    row_matrix, col_matrix, *_ = next(general_sum_data_stream)

    equilibria = week02.support_enumeration(row_matrix, col_matrix)
    equilibria = sorted(equilibria, key=lambda x: (x[0].tobytes(), x[1].tobytes()))

    for row_strategy, col_strategy in equilibria:
        assert row_strategy.dtype == np.float64, 'Incorrect dtype!'
        assert col_strategy.dtype == np.float64, 'Incorrect dtype!'

    ndarrays_regression.check(
        {f'{i}': np.concatenate(x) for i, x in enumerate(equilibria)},
        f'{request.node.originalname}{request.node.callspec.indices["general_sum_data_stream"]}',
    )
