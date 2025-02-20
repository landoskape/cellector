from typing import Optional, Union
import numpy as np
from .utils import broadcastable


def window(image: np.ndarray, kernel: Optional[Union[str, callable]] = "hanning"):
    """Apply a windowing kernel to an image.

    Parameters
    ----------
    image : np.ndarray
        The image to be windowed. Must have at least 2 dimensions. The windowing will be
        applied to the last two dimensions of the image.
    kernel : np.ndarray or callable (optional, default is "hanning")
        If 2D array, will apply to the image directly (the dimensions must match the
        last two dimensions of the image).
        If 1D array, will take the outer product of the array to create a 2D window and
        apply that to the image. In this case, the image must be square and the kernel
        must be the same size as the image.
        If callable (e.g., numpy.hanning), it will be used to create a 2D window kernel
        by applying it to the last two dimensions (e.g. height_kernel = kernel(height)).
        Then, the outer product of the height and width kernels will be applied to the
        image.

    Returns
    -------
    np.ndarray
        The windowed image.
    """
    if callable(kernel):
        height, width = image.shape[-2:]
        height_kernel = kernel(height)
        width_kernel = kernel(width)
        kernel_2d = np.outer(height_kernel, width_kernel)
    elif isinstance(kernel, np.ndarray):
        if kernel.ndim == 1:
            kernel_2d = np.outer(kernel, kernel)
        elif kernel.ndim == 2:
            kernel_2d = kernel
        else:
            raise ValueError("Array window must be 1D or 2D")
    else:
        raise TypeError("window_func must be callable or numpy array")

    if not broadcastable(kernel_2d, image):
        raise ValueError("Window and image must be broadcastable")

    return kernel_2d * image


def butterworth_bpf(
    image: np.ndarray,
    lowcut: Union[float, None],
    highcut: Union[float, None],
    order: float = 1.0,
):
    """Filter an image using a Butterworth bandpass filter.

    This function filters the image in frequency space using by applying a Butterworth
    transfer function to the image's Fourier transform. By setting either lowcut or
    highcut to None, a highpass or lowpass filter can be achieved, respectively. The
    units of the cutoff frequencies are in pixels.

    Parameters
    ----------
    image : np.ndarray
        The image to be filtered. Must have at least 2 dimensions. The filtering will be
        applied to the last two dimensions of the image.
    lowcut : float or None
        The lowcut frequency for the bandpass filter. If None, will not filter out low
        frequencies. In units of pixels.
    highcut : float or None
        The highcut frequency for the bandpass filter. If None, will not filter out high
        frequencies. In units of pixels.
    order : float (optional, default is 1)
        The order of the Butterworth filter.

    Returns
    -------
    np.ndarray
        The filtered image.
    """

    def transfer_function(freq, cutoff, order):
        """Implement transfer function for Butterworth filter."""
        return 1 / (1 + (freq / cutoff) ** (2 * order))

    # Check for valid parameters and establish filter mode
    highpass = lowcut is not None
    lowpass = highcut is not None
    if highpass and lowpass:
        if lowcut > highcut:
            raise ValueError(
                "highcut frequency should be greater than lowcut frequency"
            )
    if highpass and lowcut < 0:
        raise ValueError("frequencies must be positive")
    if lowpass and highcut < 0:
        raise ValueError("frequencies must be positive")
    if not highpass and not lowpass:
        raise ValueError("must specify either highcut or lowcut frequency")

    ny, nx = image.shape[-2:]  # Image dimensions

    # Establish the frequency grid using extra points to ensure symmetric and high resolution frequency spacing
    ndfty = 2 * ny + 1
    ndftx = 2 * nx + 1

    yfreq = np.fft.fftshift(np.fft.fftfreq(ndfty, 1 / ny))
    xfreq = np.fft.fftshift(np.fft.fftfreq(ndftx, 1 / nx))
    freq = np.sqrt(
        yfreq.reshape(-1, 1) ** 2 + xfreq.reshape(1, -1) ** 2
    )  # 2-D dft frequencies corresponds to fftshifted fft2 output

    # Create gain map over frequencies
    frequency_gain = np.ones_like(freq)
    if highpass:
        frequency_gain *= 1 - transfer_function(freq, lowcut, order)
    if lowpass:
        frequency_gain *= transfer_function(freq, highcut, order)

    # Apply the filter in the frequency domain
    fft_image = np.fft.fftshift(np.fft.fft2(image, (ndfty, ndftx)), axes=(-2, -1))
    return np.fft.ifft2(
        np.fft.ifftshift(frequency_gain * fft_image, axes=(-2, -1)), axes=(-2, -1)
    )[..., :ny, :nx].real


# Definition of the required and optional parameters for each filter method
REQUIRED_PARAMETERS = dict(
    window=[],
    butterworth_bpf=["lowcut", "highcut"],
)
OPTIONAL_PARAMETERS = dict(
    window=["kernel"],
    butterworth_bpf=["order", "fs"],
)
# Definition of which callable function to use for each filter method
FILTER_FUNCTION = dict(
    window=window,
    butterworth_bpf=butterworth_bpf,
)

# Validate filter registry
for name in FILTER_FUNCTION:
    if name not in REQUIRED_PARAMETERS:
        raise ValueError(f"Missing required parameters for filter method '{name}'")
    if name not in OPTIONAL_PARAMETERS:
        raise ValueError(f"Missing optional parameters for filter method '{name}'")
    if not callable(FILTER_FUNCTION[name]):
        raise ValueError(f"Filter method '{name}' must be callable")


def _check_parameters(name, parameters):
    """Check if the parameters are valid for the given method."""
    required = REQUIRED_PARAMETERS[name]
    optional = OPTIONAL_PARAMETERS[name]
    for key in parameters:
        if key not in required and key not in optional:
            return False, f"Invalid parameter '{key}' for filter method '{name}'"
    for key in required:
        if key not in parameters:
            return (
                False,
                f"Required parameter '{key}' missing for filter method '{name}'",
            )
    return True, None


def filter(image, name, **parameters):
    """Filter an image using a particular method with parameters specific to that method.

    The filter function is a one-size fits all function that can call any filter method
    registered in this module. The filtering method is specified by the `name` parameter,
    and any parameters required for each method are passed via the `parameters` dictionary.

    Each filtering method uses different parameters, some of which are optional and some
    of which are required.

    Parameters
    ----------
    image : np.ndarray
        The image to be filtered.
    name : str
        The name of the filtering method to use.
    instructions : dict
        A dictionary containing the parameters for the filtering method specified in `name`.

    Returns
    -------
    np.ndarray
        The filtered image.

    Examples
    --------
    >>> hanning_windowed = filter(image, "window", kernel="hanning")
    >>> custom_windowed = filter(image, "window", kernel=custom_1D_window_kernel)
    """
    if name not in FILTER_FUNCTION:
        raise ValueError(f"Filter method '{name}' not found")

    valid, message = _check_parameters(name, parameters)
    if not valid:
        raise ValueError(message)

    filter_func = FILTER_FUNCTION[name]
    return filter_func(image, **parameters)
