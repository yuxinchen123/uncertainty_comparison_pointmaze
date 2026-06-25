"""Tests for rnd_exploration.common.format: to_numpy / to_tensor / to_numpy_flat."""
import numpy as np
import pytest
import torch

from rnd_exploration.common.format import to_numpy, to_tensor, to_numpy_flat

# all tensor work stays on CPU so the suite runs in milliseconds
CPU = torch.device("cpu")


def test_to_numpy_from_tensor_roundtrip():
    """to_numpy converts a torch tensor to an equal numpy array preserving shape and values."""
    # golden path: 2D float tensor -> numpy ndarray with identical shape and values
    t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    arr = to_numpy(t)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 3)
    np.testing.assert_array_equal(arr, np.arange(6, dtype=np.float32).reshape(2, 3))
    # edge case: an already-numpy input passes through to an ndarray with the same values
    src = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = to_numpy(src)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, src)


def test_to_numpy_from_python_scalar_and_list():
    """to_numpy wraps plain Python scalars/lists via np.asarray (edge cases, no torch involved)."""
    # edge case: a Python float becomes a 0-d numpy array carrying the value
    s = to_numpy(3.5)
    assert isinstance(s, np.ndarray)
    assert s.shape == ()
    assert float(s) == pytest.approx(3.5)
    # golden path for the asarray branch: a nested list becomes a 2D ndarray
    lst = to_numpy([[1, 2], [3, 4]])
    assert lst.shape == (2, 2)
    np.testing.assert_array_equal(lst, np.array([[1, 2], [3, 4]]))


def test_to_numpy_does_not_share_dtype_change():
    """to_numpy keeps the tensor's own dtype rather than forcing float32."""
    # golden path: an int64 tensor must come back as an int numpy array, not float
    t = torch.tensor([1, 2, 3], dtype=torch.int64)
    arr = to_numpy(t)
    assert arr.dtype == np.int64
    np.testing.assert_array_equal(arr, np.array([1, 2, 3], dtype=np.int64))


def test_to_tensor_from_numpy_float32_on_device():
    """to_tensor turns a numpy array into a float32 torch tensor on the requested device."""
    # golden path: int numpy input is cast to float32 and placed on the CPU device
    src = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    t = to_tensor(src, CPU)
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.float32
    assert t.device.type == "cpu"
    assert t.shape == (2, 3)
    np.testing.assert_array_equal(t.numpy(), src.astype(np.float32))


def test_to_tensor_from_tensor_recasts_to_float32():
    """to_tensor accepts an existing tensor and returns it as float32 on the device (edge case)."""
    # edge case: a float64 tensor input is moved/recast to float32 on the device
    t_in = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    t_out = to_tensor(t_in, CPU)
    assert t_out.dtype == torch.float32
    assert t_out.device.type == "cpu"
    np.testing.assert_allclose(t_out.numpy(), np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_to_tensor_roundtrip_with_to_numpy():
    """numpy -> to_tensor -> to_numpy returns the original values (round-trip closure)."""
    # golden path: values survive the conversion cycle (float32 precision)
    src = np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
    back = to_numpy(to_tensor(src, CPU))
    assert back.shape == (3, 4)
    np.testing.assert_allclose(back, src, rtol=0, atol=1e-7)


def test_to_numpy_flat_flattens_2d_to_1d():
    """to_numpy_flat ravels a 2D tensor to a 1D float32 numpy array in row-major order."""
    # golden path: shape (2, 3) -> (6,), values in row-major order, dtype float32
    # before: [[0,1,2],[3,4,5]]  after: [0,1,2,3,4,5]
    t = torch.arange(6, dtype=torch.int64).reshape(2, 3)
    flat = to_numpy_flat(t)
    assert isinstance(flat, np.ndarray)
    assert flat.dtype == np.float32
    assert flat.shape == (6,)
    np.testing.assert_array_equal(flat, np.arange(6, dtype=np.float32))
    # column-vector edge case: shape (4, 1) -> (4,)
    col = to_numpy_flat(np.array([[1.0], [2.0], [3.0], [4.0]]))
    assert col.shape == (4,)
    np.testing.assert_array_equal(col, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))


def test_to_numpy_flat_from_numpy_input():
    """to_numpy_flat handles a numpy array input (non-tensor branch) and casts to float32."""
    # golden path: an int numpy array is flattened and cast to float32
    src = np.array([[10, 20], [30, 40]], dtype=np.int64)
    flat = to_numpy_flat(src)
    assert flat.dtype == np.float32
    assert flat.shape == (4,)
    np.testing.assert_array_equal(flat, np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32))


def test_to_numpy_flat_scalar_edge_case():
    """to_numpy_flat turns a 0-d scalar into a length-1 float32 array (edge case)."""
    # edge case: a Python scalar -> shape (1,), and a 0-d tensor -> shape (1,)
    s = to_numpy_flat(7)
    assert s.shape == (1,)
    assert s.dtype == np.float32
    assert s[0] == pytest.approx(7.0)
    t0 = to_numpy_flat(torch.tensor(2.5))
    assert t0.shape == (1,)
    assert t0[0] == pytest.approx(2.5)
