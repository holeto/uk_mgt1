import hashlib
import sys
from collections.abc import Generator

sys.path.append('solutions')
sys.path.append("../")

import numpy as np
import pytest
from pytest import FixtureRequest
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

import templates.week04 as week04
from utils import (
    cache_data_stream_per_function,
    parameterize_random_general_sum_tests,
    parameterize_random_zero_sum_tests,
)

BASE_SEED = 141


@pytest.fixture
def rng(request: FixtureRequest) -> np.random.Generator:
    """Create a random seed based on the function name to ensure reproducibility."""

    digest = hashlib.md5(request.node.originalname.encode()).hexdigest()
    seed = (BASE_SEED + int(digest, 16)) % (2**32)

    return np.random.default_rng(seed)


@pytest.fixture
@cache_data_stream_per_function
def general_sum_data_stream(request: FixtureRequest, rng: np.random.Generator) -> Generator:
    return parameterize_random_general_sum_tests(15, rng)


@pytest.fixture
@cache_data_stream_per_function
def zero_sum_data_stream(request: FixtureRequest, rng: np.random.Generator) -> Generator:
    return parameterize_random_zero_sum_tests(15, rng)


@pytest.mark.parametrize('zero_sum_data_stream', range(5), indirect=True)
def test_find_nash_equilibrium(
    zero_sum_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
) -> None:
    row_matrix, *_ = next(zero_sum_data_stream)

    row_strategy, col_strategy = week04.find_nash_equilibrium(row_matrix)

    assert row_strategy.dtype == np.float64, 'Incorrect dtype!'
    assert col_strategy.dtype == np.float64, 'Incorrect dtype!'

    ndarrays_regression.check(
        {'row_strategy': row_strategy, 'col_strategy': col_strategy},
        f'{request.node.originalname}{request.node.callspec.indices["zero_sum_data_stream"]}',
    )


@pytest.mark.parametrize('zero_sum_data_stream', range(5), indirect=True)
def test_find_correlated_equilibrium_zero_sum(
    zero_sum_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
) -> None:
    row_matrix, col_matrix, *_ = next(zero_sum_data_stream)

    corr_equi = week04.find_correlated_equilibrium(row_matrix, col_matrix)

    assert corr_equi.dtype == np.float64, 'Incorrect dtype!'

    ndarrays_regression.check(
        {'correlated_equilibrium': corr_equi},
        f'{request.node.originalname}{request.node.callspec.indices["zero_sum_data_stream"]}',
    )


@pytest.mark.parametrize('general_sum_data_stream', range(5), indirect=True)
def test_find_correlated_equilibrium_general_sum(
    general_sum_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
) -> None:
    row_matrix, col_matrix, *_ = next(general_sum_data_stream)

    corr_equi = week04.find_correlated_equilibrium(row_matrix, col_matrix)

    assert corr_equi.dtype == np.float64, 'Incorrect dtype!'

    ndarrays_regression.check(
        {'correlated_equilibrium': corr_equi},
        f'{request.node.originalname}{request.node.callspec.indices["general_sum_data_stream"]}',
    )
