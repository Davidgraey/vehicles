import numpy as np
from numpy.typing import NDArray

EPSILON = 1e-15

def layer_norm(x: NDArray) -> NDArray:
    """
    layer norm - across -1 dimension (embedding space) - center mean and
        variance for each sample, for each time step.
    Parameters
    ----------
    x : array of data to normalize on last axis (embedding dim)

    Returns
    -------
    x', normalized x data
    """
    dimensionality = x.ndim
    if dimensionality <= 3:
        axis=-1
    else:
        raise ValueError("too many dims in the layer_norm process")
    # zero mean and variance --
    _var = np.var(x, axis=axis, keepdims=True)
    _mean = np.mean(x, axis=axis, keepdims=True)
    _x = (x - _mean) / (np.sqrt(_var + EPSILON))
    return _x


def fft_forward(input_data: NDArray) -> NDArray:
    # (num_samples, sequence_length, embedding_dimension)
    fourier_transfrorm = np.real(np.fft.fft2(input_data, axes=(-1, -2)))
    return layer_norm(input_data + fourier_transfrorm)

