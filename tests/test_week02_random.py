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
    return parameterize_random_general_sum_tests(8, rng)


@pytest.fixture
@cache_data_stream_per_function
def zero_sum_data_stream(request: FixtureRequest, rng: np.random.Generator) -> Generator:
    return parameterize_random_zero_sum_tests(8, rng)


@pytest.mark.parametrize('general_sum_data_stream', range(5), indirect=True)
def test_support_enumeration_general_sum(
    general_sum_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
) -> None:
    row_matrix, col_matrix, *_ = next(general_sum_data_stream)

    equilibria = week02.support_enumeration(row_matrix, col_matrix)
    equilibria = sorted(equilibria, key=lambda x: (x[0].tobytes(), x[1].tobytes()))

    ndarrays_regression.check(
        {f'{i}': np.concatenate(x) for i, x in enumerate(equilibria)},
        f'{request.node.originalname}{request.node.callspec.indices["general_sum_data_stream"]}',
    )


@pytest.mark.parametrize('zero_sum_data_stream', range(5), indirect=True)
def test_support_enumeration_zero_sum(
    zero_sum_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
) -> None:
    row_matrix, col_matrix, *_ = next(zero_sum_data_stream)

    equilibria = week02.support_enumeration(row_matrix, col_matrix)
    equilibria = sorted(equilibria, key=lambda x: (x[0].tobytes(), x[1].tobytes()))

    ndarrays_regression.check(
        {f'{i}': np.concatenate(x) for i, x in enumerate(equilibria)},
        f'{request.node.originalname}{request.node.callspec.indices["zero_sum_data_stream"]}',
    )
