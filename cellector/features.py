from typing import List
import numpy as np
from .utils import center_surround, cross_power_spectrum


def phase_correlation_zero(centered_masks: np.ndarray, centered_reference: np.ndarray, eps: float = 1e6):
    """Measure the zero-offset phase correlation between two images.

    Computes the phase correlation in the fourier domain with the following formula:
    .. math::
        R = F^{-1}(F(static\_image) * conj(F(moving\_image)))

    where F is the fourier transform and conj is the complex conjugate. Returns the
    zero-offset real component, which describes the phase correlation of the two images
    without any shift.

    Note: broadcasting between the two images is accepted as long as the last two
    dimensions have equal shapes, which represent the "image" dimensions.

    Parameters
    ----------
    centered_masks : np.ndarray with shape (..., M, N)
        The centered mask images for each ROI.
    centered_reference : np.ndarray with shape (..., M, N)
        The centered reference image (centered on each ROI).
    eps : float, optional
        Offset value to avoid division by zero. Default is 1e6.

    Returns
    -------
    float or np.ndarray
        The central pixel value of the phase correlation, representing the
        correlation between the masks and the reference images without any
        shift. Shape will be (...) for input images of shape (..., M, N).

    See Also
    --------
    phase_correlation : Full version that returns the entire correlation map.
    """
    R = cross_power_spectrum(centered_masks, centered_reference, eps=eps)
    center_value = np.sum(R, axis=(-2, -1)) / (R.shape[-2] * R.shape[-1])
    return center_value.real


def in_vs_out(centered_masks: np.ndarray, centered_reference: np.ndarray, iterations: int = 7):
    """Measure the ratio of the intensity inside the mask to the local intensity surrounding the mask.

    The intensity inside the mask is the sum of the reference image inside the mask, and the
    intensity surrounding the mask is the sum of the reference image inside and outside the mask.
    The ratio of these two values is a measure of how much the mask is pointing toward intensity
    features in the reference image.

    Note: this is the closest feature to the native suite2p red cell probability feature. So, in the case
    where the suite2p red cell probability isn't available, this is a good alternative (and it might be
    useful on it's own). The suite2p method uses a similar concept but the surround region is based on the
    neuropil model rather than a footprint as in this function.

    Parameters
    ----------
    centered_masks : np.ndarray with shape (..., M, N)
        The centered mask images for each ROI.
    centered_reference : np.ndarray with shape (..., M, N)
        The centered reference image (centered on each ROI).
    iterations : int, optional
        The number of iterations to dilate the mask for the surround. Default is 7.

    Returns
    -------
    np.ndarray
        The ratio of the intensity inside the mask to the intensity inside plus outside the mask.
        Shape will be (N,), where N is the number of ROIs.
    """
    center, surround = center_surround(centered_masks, iterations=iterations)
    inside_sum = np.sum(center * centered_reference, axis=(-2, -1))
    outside_sum = np.sum(surround * centered_reference, axis=(-2, -1))
    return inside_sum / (inside_sum + outside_sum)


def dot_product(lam: List[np.ndarray], ypix: List[np.ndarray], xpix: List[np.ndarray], plane_idx: List, reference: np.ndarray):
    """Measure the normalized dot product between the masks and the reference images.

    Will take the inner product of the mask weights with the reference image, and divide
    by the norm of the mask weights. This is a measure of how well the mask points toward
    intensity features in the reference image.

    Uses lam, xpix, ypix instead of mask images for efficiency. This is the best function
    to use to compute the dot_product feature because it is usually faster, always uses the
    full ROI mask (whereas part of the mask can be cropped in the centered mask version called
    dot_product_array), and it requires fewer preprocessing steps.

    Parameters
    ----------
    lam : List[np.ndarray]
        List of mask weights for each ROI.
    ypix : List[np.ndarray]
        List of y-pixel indices for each ROI.
    xpix : List[np.ndarray]
        List of x-pixel indices for each ROI.
    plane_idx : List
        idx to reference image for each ROI.
    reference : List[np.ndarray]
        The reference image.

    Returns
    -------
    np.ndarray
        The dot product between the masks and the reference images. Shape will be (N,),
        where N is the number of ROIs.

    See Also
    --------
    dot_product_array : Function that uses mask images instead of weights and pixel indices.
    """
    # TODO: consider optimizing with multiprocessing for example.
    dot_product = np.zeros(len(lam))
    for idx, (l, x, y, p) in enumerate(zip(lam, xpix, ypix, plane_idx)):
        # reference image for this ROI
        ref = reference[p]
        # dot product between mask and reference image
        dot = np.sum(ref[y, x] @ l / np.linalg.norm(l))
        # store dot product
        dot_product[idx] = dot
    return dot_product


def dot_product_array(centered_masks: np.ndarray, centered_reference: np.ndarray):
    """Measure the normalized dot product between the masks and the reference images.

    Will take the inner product of the mask weights with the reference image, and divide
    by the norm of the mask weights. This is a measure of how well the mask points toward
    intensity features in the reference image.

    Parameters
    ----------
    centered_masks : np.ndarray with shape (..., M, N)
        The centered mask images for each ROI.
    centered_reference : np.ndarray with shape (..., M, N)
        The centered reference image (centered on each ROI).

    Returns
    -------
    np.ndarray
        The dot product between the masks and the reference images. Shape will be (N,),
        where N is the number of ROIs.

    See Also
    --------
    dot_product : Function that uses mask weights and pixel indices instead of images.
    """
    sum = np.sum(centered_masks * centered_reference, axis=(-2, -1))
    norm = np.linalg.norm(centered_masks, axis=(-2, -1))
    return sum / norm


def compute_correlation(centered_masks: np.ndarray, centered_references: np.ndarray):
    """Measure the correlation coefficient between the masks and the reference images.

    The masks and reference images should have the same shape and be centered on each other.
    It is expected (but not required) that the masks and references are filtered to have
    np.nan outside the surround of each ROI to focus the correlation on the local structure
    near the ROI rather than the full square provided by the centering stacks. Since the
    surround is larger than the ROI by definition, this is effectively comparing the local
    data in the ROI with the zeros outside the ROI.

    Parameters
    ----------
    centered_masks : np.ndarray with shape (..., M, N)
        The centered mask images for each ROI.
    centered_reference : np.ndarray with shape (..., M, N)
        The centered reference image (centered on each ROI).

    Returns
    -------
    np.ndarray
        The correlation coefficient between the masks and the reference images. Shape will be (N,),
        where N is the number of ROIs.

    See Also
    --------
    utils.surround_filter : Function to filter the masks and references to have np.nan outside the ROI.
    """
    u_ref = np.nanmean(centered_references, axis=(-2, -1), keepdims=True)
    u_mask = np.nanmean(centered_masks, axis=(-2, -1), keepdims=True)
    s_ref = np.nanstd(centered_references, axis=(-2, -1))
    s_mask = np.nanstd(centered_masks, axis=(-2, -1))
    N = np.sum(~np.isnan(centered_masks), axis=(-2, -1))
    return np.nansum((centered_references - u_ref) * (centered_masks - u_mask), axis=(-2, -1)) / N / s_ref / s_mask
