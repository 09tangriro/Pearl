import numpy as np
import pytest
import torch as T

from anvil.common.utils import extend_shape, numpy_to_torch, torch_to_numpy

numpy_data = (np.zeros(shape=(2, 2)), np.zeros(shape=(3, 3)))
torch_data = (T.zeros(2, 2), T.zeros(3, 3))
mixed_data = (np.zeros(shape=(2, 2)), T.zeros(3, 3))
one_numpy = np.zeros(shape=(2, 2))
one_torch = T.zeros(2, 2)


@pytest.mark.parametrize("input", [numpy_data, torch_data, mixed_data])
def test_numpy_to_torch(input):
    actual_output = numpy_to_torch(*input)
    for i in range(len(actual_output)):
        assert T.equal(actual_output[i], torch_data[i])


@pytest.mark.parametrize("input", [numpy_data, torch_data, mixed_data])
def test_torch_to_numpy(input):
    actual_output = torch_to_numpy(*input)
    for i in range(len(actual_output)):
        np.testing.assert_array_equal(actual_output[i], numpy_data[i])


def test_one_input():
    actual_output = torch_to_numpy(one_torch)
    np.testing.assert_array_equal(actual_output, one_numpy)

    actual_output = numpy_to_torch(one_numpy)
    T.equal(actual_output, one_torch)


def test_extend_shape():
    shape = (1, 1, 1)

    actual_output = extend_shape(shape, 5)
    expected_output = (5, 1, 1)

    assert actual_output == expected_output
