from typing import List, Tuple, Optional, Sequence
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import numba as nb
from scipy.ndimage import binary_dilation, generate_binary_structure


def _broadcastable(x: np.ndarray, y: np.ndarray) -> bool:
    """Check if two numpy arrays are broadcastable.

    Parameters
    ----------
    x, y : np.ndarray
        Two numpy arrays to check for broadcast compatibility.

    Returns
    -------
    bool
        True if the arrays are broadcastable, False otherwise.
    """
    xshape = reversed(x.shape)
    yshape = reversed(y.shape)
    for i, j in zip(xshape, yshape):
        if i != j and i != 1 and j != 1:
            return False
    return True


def transpose(sequence: Sequence) -> List:
    """Return a transpose of a sequence of elements in a list or tuple.

    This function helps manage transposing the order of elements in a list or tuple. For
    example, if you have a list of lists, this will return a list of lists where the first
    element of each sublist in the input becomes the elements of the first sublist of the
    output (and second elements in each list will make up the second sublist).

    Parameters
    ----------
    sequence : Sequence
        A sequence of elements to transpose. Can be a list or tuple, but each should have
        the same number of elements.

    Returns
    -------
    map of lists

    Examples
    --------
    To transpose a list of lists:
    >>> transpose([[1, 2, 3], [4, 5, 6]])
    [[1, 4], [2, 5], [3, 6]]

    If you perform a function that outputs tuples in a list comprehension, but want to
    organize the results by the elements of the tuple rather than the iterable on the
    list comprehension:
    >>> def func(x):
    ...     return x, x ** 2
    >>> results = [func(i) for i in range(3)]
    [(0, 0), (1, 1), (2, 4)]
    >>> transpose(results)
    [(0, 1, 2), (0, 1, 4)]
    """
    return map(list, zip(*sequence))


def _get_pixel_data_single(mask: np.ndarray):
    """Get pixel data from a single mask.

    Extracts the intensity values, y-coordinates, and x-coordinates from a single mask
    footprint with intensity values.

    Parameters
    ----------
    mask : np.ndarray
        A 2D mask footprint with intensity values.

    Returns
    -------
    lam, ypix, xpix : tuple of np.ndarrays
        Intensity values, y-coordinates, and x-coordinates for the mask.
    """
    ypix, xpix = np.where(mask)
    lam = mask[ypix, xpix]
    return lam, ypix, xpix


def get_pixel_data(mask_volume, as_stats: bool = True, verbose: bool = True):
    """Get pixel data from a mask volume.

    Extracts the intensity values, y-coordinates, and x-coordinates from a mask volume
    where each slice of the volume corresponds to a single ROI.

    Parameters
    ----------
    mask_volume : np.ndarray
        A 3D mask volume with shape (num_rois, height, width) where each slice is a mask
        footprint with intensity values.
    as_stats : bool, optional
        Whether to return the data as a list of dictionaries with keys "lam", "ypix", and "xpix"
        or to return a tuple of lists for each data type. Default is True.
    verbose : bool, optional
        Whether to use a tqdm progress bar to show progress. Default is True.

    Returns
    -------
    lam, ypix, xpix : tuple of list of np.ndarrays
        Intensity values, y-coordinates, and x-coordinates for each ROI.

    or if as_stats is True:

    stats : list of dict
        List of dictionaries with keys "lam", "ypix", and "xpix" for each ROI.
    """
    iterable = tqdm(mask_volume, desc="Extracting mask data", leave=False) if verbose else mask_volume
    with Pool(max(2, cpu_count() - 2)) as pool:
        results = pool.map(_get_pixel_data_single, iterable)
    lam, ypix, xpix = transpose(results)
    if as_stats:
        return [dict(lam=l, ypix=y, xpix=x) for l, y, x in zip(lam, ypix, xpix)]
    return lam, ypix, xpix


def get_s2p_data(s2p_dir: Path, reference_key: str = "meanImg_chan2"):
    """Get list of stats and chan2 reference images from all planes in a suite2p directory.

    suite2p saves the statistics and reference images for each plane in separate
    directories. This function reads the statistics and reference images for each plane
    and returns them as lists.

    Parameters
    ----------
    s2p_dir : Path
        Path to the suite2p directory, which contains directories for each plane in the
        format "plane0", "plane1", etc.
    reference_key : str, optional
        Key to use for the reference image in the ops dictionary. Default is "meanImg_chan2".

    Returns
    -------
    stats : list of list of dictionaries
        Each element of stats is a list of dictionaries containing ROI statistics for each plane.
    references : list of np.ndarrays
        Each element of references is an image (usually of average red fluorescence) for each plane.
    """
    planes = s2p_dir.glob("plane*")
    stats = []
    references = []
    for plane in planes:
        stats.append(np.load(plane / "stat.npy", allow_pickle=True))
        ops = np.load(plane / "ops.npy", allow_pickle=True).item()
        references.append(ops[reference_key])
    if not all(ref.shape == references[0].shape for ref in references):
        raise ValueError("Reference images must have the same shape as each other!")
    if not all(ref.ndim == 2 for ref in references):
        raise ValueError("Reference images must be 2D arrays!")
    return stats, references


def get_s2p_redcell(s2p_dir: Path):
    """Get red cell probability masks from all planes in a suite2p directory.

    Extracts the red cell probability masks from each plane in a suite2p directory
    and returns them as a list of numpy arrays. The red cell probability masks are
    saved in the "redcell.npy" file in each plane directory in which the first column
    is a red cell assigment and the second column is the probability of each ROI being
    a red cell.

    Parameters
    ----------
    s2p_dir : Path
        Path to the suite2p directory, which contains directories for each plane in the
        format "plane0", "plane1", etc.

    Returns
    -------
    redcell : list of np.ndarrays
        List of red cell probabilities for each plane. Each array has length N corresponding
        to the number of ROIs in that plane.
    """
    planes = s2p_dir.glob("plane*")
    redcell = []
    for plane in planes:
        c_redcell = np.load(plane / "redcell.npy")
        redcell.append(c_redcell[:, 1])
    return redcell


def get_roi_data(stat: List[dict]):
    """Get ROI data from a list of stat objects.

    stat is a list of dictionaries describing the statistics of ROIs. Just like the
    standard output of suite2p, the dictionaries should contain the following keys:
    - "lam": intensity values for each pixel in the ROI
    - "ypix": y-coordinates of pixels in the ROI
    - "xpix": x-coordinates of pixels in the ROI

    This function is simply a convenience function to extract the data from the list
    of dictionaries and return them as separate lists for each key.

    Parameters
    ----------
    stat : list of dict
        List of dictionaries containing ROI data (including "lam", "ypix", "xpix").

    Returns
    -------
    lam, ypix, xpix : tuple of list
        Lists of intensity values, y-coordinates, and x-coordinates for each ROI.
    """
    lam = [s["lam"] for s in stat]
    ypix = [s["ypix"] for s in stat]
    xpix = [s["xpix"] for s in stat]
    return lam, ypix, xpix


def cat_planes(list_of_sequences: List[Sequence]):
    """Concatenate a sequence of sequences into a single sequence.

    Parameters
    ----------
    sequence : list of sequences
        A list of sequences to concatenate.

    Returns
    -------
    list or numpy.ndarray
        A single list containing all elements from the input sequences. If all the input
        sequences are numpy arrays, the output will be a single numpy array.

    Notes
    -----
    This function is used to concatenate list of ROI data from across planes
    because we can optimize operations by performing them on all ROIs at once
    rather than performing them on each plane separately. To return to lists
    of each plane independently, use the split_planes function.
    """
    if all(isinstance(seq, np.ndarray) for seq in list_of_sequences):
        return np.concatenate(list_of_sequences)
    return [item for subsequence in list_of_sequences for item in subsequence]


def split_planes(sequence: Sequence, num_per_plane: List[int]):
    """
    Split a sequence into a list of sequences based on the number of elements per plane.

    Parameters
    ----------
    sequence : list
        A single sequence to split into multiple sequences.
    num_per_plane : list
        A list of integers indicating the number of elements per plane.

    Returns
    -------
    list
        A list of sequences, each containing the elements for a single plane.

    Notes
    -----
    This function is used to split concatenated ROI data back into separate
    lists for each plane after operations have been performed on all ROIs at once
    using the cat_planes function.
    """
    return [sequence[sum(num_per_plane[:i]) : sum(num_per_plane[: i + 1])] for i in range(len(num_per_plane))]


def flatten_roi_data(lam: List[np.ndarray], ypix: List[np.ndarray], xpix: List[np.ndarray]):
    """Get flattened ROI data for use in parallel processing.

    Returns flattened arrays for intensity values, y-coordinates, x-coordinates in which
    the data is concatenated across ROIs such that each is a single numpy array. The ROI
    index is also returned as a single numpy array with the same length as the other arrays
    to indicate which ROI each value belongs to.

    Parameters
    ----------
    lam : list of np.ndarrays
        Intensity values for each pixel in each ROI.
    ypix : list of np.ndarrays
        Y-coordinates of pixels in each ROI.
    xpix : list of np.ndarrays
        X-coordinates of pixels in each ROI.

    Returns
    -------
    lam_flat, ypix_flat, xpix_flat, roi_idx : tuple of np.ndarrays
        Flattened arrays for intensity values, y-coordinates, x-coordinates, and ROI index
        associated with each lam/y/x value.
    """
    lam_flat = np.concatenate(lam)
    ypix_flat = np.concatenate(ypix)
    xpix_flat = np.concatenate(xpix)
    roi_idx = np.repeat(np.arange(len(lam)), [len(arr) for arr in lam])
    return lam_flat, ypix_flat, xpix_flat, roi_idx


def get_roi_centroid(lam: np.ndarray, ypix: np.ndarray, xpix: np.ndarray, method: str = "weightedmean", asint: bool = True):
    """Calculate the centroid coordinates of an ROI.

    Parameters
    ----------
    lam : np.ndarray
        Intensity values for each pixel in the ROI.
    ypix : np.ndarray
        Y-coordinates of pixels in the ROI.
    xpix : np.ndarray
        X-coordinates of pixels in the ROI.
    method : {'weightedmean', 'median'}, optional
        Method to calculate the centroid:
        - 'weightedmean': weighted average using pixel intensities (default)
        - 'median': median position of pixels (independent for y & x pixels)
    asint : bool, optional
        Whether to return the centroid as an integer (default).
        If set to False, will return a float when method="weightedmean".

    Returns
    -------
    yc, xc : tuple of int or float
        Y and X coordinates of the centroid. Returns as integers by default,
        but will return floats if asint=False and method="weightedmean".
    """
    if method == "weightedmean":
        yc = np.sum(lam * ypix) / np.sum(lam)
        xc = np.sum(lam * xpix) / np.sum(lam)
        if asint:
            yc, xc = int(yc), int(xc)
    elif method == "median":
        yc = int(np.median(ypix))
        xc = int(np.median(xpix))
    else:
        raise ValueError(f"Invalid method ({method}). Must be 'weightedmean' or 'median'")
    return yc, xc


def get_roi_centroids(lam: List[np.ndarray], ypix: List[np.ndarray], xpix: List[np.ndarray], method: str = "weightedmean", asint: bool = True):
    """Get the centroid of each ROI in a list of ROIs.

    Parameters
    ----------
    lam : list of np.ndarrays
        Intensity values for each pixel in each ROI.
    ypix : list of np.ndarrays
        Y-coordinates of pixels in each ROI.
    xpix : list of np.ndarrays
        X-coordinates of pixels in each ROI.
    method : {'weightedmean', 'median'}, optional
        Method to calculate the centroid:
        - 'weightedmean': weighted average using pixel intensities (default)
        - 'median': median position of pixels (independent for y & x pixels)
    asint : bool, optional
        Whether to return the centroid as an integer (default).
        If set to False, will return a float when method="weightedmean".

    Returns
    -------
    yc, xc : tuple of np.ndarrays of int or float
        Y and X coordinates of each ROI centroid. Returns as integers by default,
        but will return floats if asint=False and method="weightedmean".

    See Also
    --------
    get_roi_centroid : The underlying function for centroid calculation.
    """
    centroids = [get_roi_centroid(l, y, x, method=method, asint=asint) for (l, y, x) in zip(lam, ypix, xpix)]
    yc, xc = transpose(centroids)
    return yc, xc


def get_mask_volume(lam: np.ndarray, ypix: np.ndarray, xpix: np.ndarray, roi_idx: np.ndarray, num_rois: int, shape: Tuple[int, int]):
    """Create a mask stack using Numba for parallel processing.

    For optimization with numba, the input is flattened arrays of pixel coordinates and intensities
    concatenated across ROIs, with the ROI index indicated by ``roi_idx``. The volume is created by
    assigning the intensity values to the appropriate pixel coordinates in the mask stack with zeros
    everywhere else. Each slice of the mask stack corresponds to a single ROI.

    Parameters
    ----------
    lam : np.ndarray
        Intensity values for each pixel in each ROI.
        In format (num_pixels,) with values from each ROI concatenated with each other.
    ypix : np.ndarray
        Y-coordinates of pixels in each ROI.
        Same format as lam.
    xpix : np.ndarray
        X-coordinates of pixels in each ROI.
        Same format as lam.
    roi_idx : np.ndarray
        ROI index for each value in lam, ypix, xpix
    num_rois : int
        Number of ROIs described in the input arrays.
    shape : tuple of int
        Shape of the mask stack to create (height and width).

    Returns
    -------
    np.ndarray
        A 3D mask stack with shape (num_rois, height, width) where the mask footprints
        are filled in with the intensity values (lam) and zeros elsewhere.
    """
    mask_stack = np.zeros((num_rois, *shape), dtype=np.float32)
    return _nb_get_mask_volume(lam, ypix, xpix, roi_idx, mask_stack)


@nb.njit(parallel=True)
def _nb_get_mask_volume(lam, ypix, xpix, roi_idx, mask_stack):
    for idx in nb.prange(len(lam)):
        mask_stack[roi_idx[idx], ypix[idx], xpix[idx]] = lam[idx]
    return mask_stack


# # Might use in a nonumba mode, dispatched by make_mask_stack
# def _nonumba_mask_stack(lam, ypix, xpix, shape):
#     """Create a 3D mask stack from lists of pixel coordinates and intensities."""
#     mask_stack = np.zeros((len(lam), *shape), dtype=np.float32)
#     for idx, (l, y, x) in enumerate(zip(lam, ypix, xpix)):
#         mask_stack[idx, y, x] = l
#     return mask_stack


def get_centered_masks(
    lam: np.ndarray,
    ypix: np.ndarray,
    xpix: np.ndarray,
    roi_idx: np.ndarray,
    centroids: Tuple[np.ndarray, np.ndarray],
    width: Optional[int] = 15,
    fill_value: Optional[float] = 0.0,
):
    """Create a stack of centered masks for each ROI.

    For optimization with numba, the input is flattened arrays of pixel coordinates and intensities
    concatenated across ROIs, with the ROI index indicated by ``roi_idx``. The centroids are provided
    as a tuple of y and x coordinates for each ROI. The mask stack is created by centering each mask
    around the centroid of the ROI, with a width of ``width`` pixels on each side. Any pixels outside
    the mask footprint are filled with the value specified by ``fill_value``.

    Parameters
    ----------
    lam : np.ndarray
        Intensity values for each pixel in each ROI.
        In format (num_pixels,) with values from each ROI concatenated with each other.
    ypix : np.ndarray
        Y-coordinates of pixels in each ROI.
        Same format as lam.
    xpix : np.ndarray
        X-coordinates of pixels in each ROI.
        Same format as lam.
    roi_idx : np.ndarray
        ROI index for each value in lam, ypix, xpix
    centroids : Tuple[np.ndarray, np.ndarray]
        Tuple of y and x centroids for each ROI.
    width : int, optional
        Width in pixels around the ROI centroid. Default is 15.
    fill_value : float, optional
        Value to use as the background. Default is 0.0.

    Returns
    -------
    np.ndarray
        A 3D stack of centered masks for each ROI with shape (num_rois, 2 * width + 1, 2 * width + 1).
    """
    yc, xc = centroids
    mask_stack = _nb_get_centered_masks(lam, ypix, xpix, roi_idx, yc, xc, width, fill_value)
    return mask_stack


@nb.njit(parallel=True)
def _nb_get_centered_masks(lam, ypix, xpix, roi_idx, yc, xc, width, fill_value):
    """Create a stack of centered masks for each ROI using Numba for parallel processing."""
    mask_stack = np.full((len(yc), 2 * width + 1, 2 * width + 1), fill_value, dtype=np.float32)
    for idx in nb.prange(len(lam)):
        c_yc, c_xc = yc[roi_idx[idx]], xc[roi_idx[idx]]
        cyidx = ypix[idx] - c_yc + width
        cxidx = xpix[idx] - c_xc + width
        if cyidx >= 0 and cyidx < 2 * width + 1 and cxidx >= 0 and cxidx < 2 * width + 1:
            mask_stack[roi_idx[idx], cyidx, cxidx] = lam[idx]
    return mask_stack


# Might use in a nonumba mode, dispatched by centered_mask_stack
# def _nonumba_get_centered_masks(lam, ypix, xpix, centroids, width=15, fill_value=0.0):
#     """Create a stack of centered masks for each ROI."""
#     yc, xc = centroids
#     num_rois = len(lam)
#     mask_stack = np.full((num_rois, 2 * width + 1, 2 * width + 1), fill_value, dtype=np.float32)
#     for idx, (yc_i, xc_i) in enumerate(zip(yc, xc)):
#         cyidx = ypix[idx] - yc_i + width
#         cxidx = xpix[idx] - xc_i + width
#         idx_use_points = (cyidx >= 0) & (cyidx < 2 * width + 1) & (cxidx >= 0) & (cxidx < 2 * width + 1)
#         mask_stack[idx, cyidx[idx_use_points], cxidx[idx_use_points]] = lam[idx][idx_use_points]
#     return mask_stack


def _get_roi_bounds(center: int, width: int, image_dim: int):
    """Get the start and end indices for a centered ROI

    Returns the start and end indices for a centered ROI around a given center
    and a given number of pixels to each side. Also returns the offsets for the
    start and end indices to account for edge cases where less than width
    pixels are available on one side.

    Parameters
    ----------
    center : int
        The center of the ROI.
    width : int
        The number of pixels to each side around the center.
    image_dim : int
        The dimension of the image to draw from.

    Returns
    -------
    start, end, start_offset, end_offset : tuple of ints
        The start and end indices for the ROI and the offsets for the start and end indices.

    Examples
    --------
    To get the bounds for a centered ROI with 15 pixels to each side
    around a center of 5, in an image of size 20:
    >>> _get_roi_bounds(5, 15, 20)
    (0, 20, 10, -1)

    Then to insert a centered patch into an array:
    (Usually the patch is an image and the bounds are measured for x & y dimensions separately)
    >>> start, end, start_offset, end_offset = _get_roi_bounds(10, 5, 20)
    >>> slice = slice(start_offset, 2 * 5 + 1 + end_offset)
    >>> array[slice] = patch[start:end]
    """
    if width < 0 or not isinstance(width, int):
        raise ValueError("Width must be a non-negative integer")
    if not isinstance(center, int):
        raise ValueError("Center must be an integer")
    if not isinstance(image_dim, int):
        raise ValueError("Image dimension must be an integer")

    # Get raw start and end indices (might be negative or larger than image_dim)
    start_raw = center - width
    end_raw = center + width + 1

    # Clip start and end indices to image dimensions
    start = max(start_raw, 0)
    end = min(end_raw, image_dim)

    # Measure the offset for the start and end indices
    start_offset = start - start_raw
    end_offset = end - end_raw

    return start, end, start_offset, end_offset


def get_centered_references(
    references: List[np.ndarray],
    plane_idx: List[int],
    centroids: Tuple[List[int]],
    width: Optional[int] = 15,
    fill_value: Optional[float] = 0.0,
):
    """Create a stack of centered reference images on each ROI.

    Parameters
    ----------
    references : List[np.ndarray]
        List of reference images for each plane.
    plane_idx : List[int]
        List of plane indices for each ROI.
    centroids : Tuple[np.ndarray, np.ndarray]
        Tuple of y and x centroids for each ROI.
    width : int, optional
        Width in pixels around the ROI centroid. Default is 15.
    fill_value : float, optional
        Value to use as the background. Default is 0.0.

    Returns
    -------
    np.ndarray
        A 3D stack of centered reference images for each ROI
        with shape (num_rois, 2 * width + 1, 2 * width + 1).
    """
    if not isinstance(width, int) or width < 0:
        raise ValueError("Width must be a non-negative integer")

    # Split centroids into y and x coordinates
    yc, xc = centroids

    # Preallocate reference stack array
    num_rois = len(plane_idx)
    ref_stack = np.full((num_rois, 2 * width + 1, 2 * width + 1), fill_value, dtype=np.float32)

    # For each ROI, fill in a patch of the appropriate reference image into the centered_stack array
    for idx in range(num_rois):
        ystart, yend, ystart_offset, yend_offset = _get_roi_bounds(yc[idx], width, references[plane_idx[idx]].shape[0])
        xstart, xend, xstart_offset, xend_offset = _get_roi_bounds(xc[idx], width, references[plane_idx[idx]].shape[1])
        y_centered_slice = slice(ystart_offset, 2 * width + 1 + yend_offset)
        x_centered_slice = slice(xstart_offset, 2 * width + 1 + xend_offset)
        ref_stack[idx, y_centered_slice, x_centered_slice] = references[plane_idx[idx]][ystart:yend, xstart:xend]
    return ref_stack


def center_surround(centered_masks: np.ndarray, iterations: Optional[int] = 7):
    """Calculate the center and surround footprints of a stack of centered masks.

    The center footprint is the binary mask of the ROI itself, while the surround
    footprint is a binary mask of the surrounding region of the ROI. The surround
    region is defined as the dilation of the center region by a 3x3 cross structuring
    element for 7 iterations.

    Parameters
    ----------
    centered_masks : np.ndarray
        The centered mask images for each ROI. Should have shape (num_rois, height, width).
    iterations : int, optional
        The number of iterations for the binary dilation. Default is 7.

    Returns
    -------
    center, surround : Tuple[np.ndarray, np.ndarray]
        The center and surround binary masks for each ROI.
    """
    # Footprint of ROI defined as any non-zero pixel
    center = np.where(centered_masks > 0, 1, 0)

    # This structuring element allows parallelization across ROIs without interference
    stack_structure = np.stack((np.zeros((3, 3), dtype=bool), generate_binary_structure(2, 1), np.zeros((3, 3), dtype=bool)))
    surround = binary_dilation(center, structure=stack_structure, iterations=iterations)

    # Return as boolean arrays for use in indices and as masks
    return center.astype(bool), surround.astype(bool)


def surround_filter(centered_masks: np.ndarray, centered_references: np.ndarray, iterations: Optional[int] = 7):
    """Filter centered masks and references to include only the surround region of each mask.

    Uses the center_surround function to calculate the center and surround footprints, which
    is based on a binary dilation of the mask footprint. The surround region is then used to
    filter the centered masks and references, filling anything outside the surround region
    with np.nan.

    Parameters
    ----------
    centered_masks : np.ndarray
        The centered mask images for each ROI. Should have shape (num_rois, height, width).
    centered_references : np.ndarray
        The centered reference images around each ROI. Should have shape (num_rois, height, width).
    iterations : int, optional
        The number of iterations for the binary dilation. Default is 7.

    Returns
    -------
    masks_surround, references_surround : Tuple[np.ndarray, np.ndarray]
        The centered mask and reference images for the surround region of each ROI.
        Anything outside the surround region will be filled with np.nan.
    """
    surround = center_surround(centered_masks, iterations=iterations)[1]
    masks_surround = np.where(surround, centered_masks, np.nan)
    references_surround = np.where(surround, centered_references, np.nan)
    return masks_surround, references_surround


def cross_power_spectrum(static_image: np.ndarray, moving_image: np.ndarray, eps: float = 1e6):
    """Measure the cross-power spectrum between two images.

    Computes the cross-power spectrum in the fourier domain with the following formula:
    .. math::
        R = F(static\_image) * conj(F(moving\_image))

    where F is the fourier transform and conj is the complex conjugate.

    Parameters
    ----------
    static_image : np.ndarray with shape (..., M, N)
        The static image.
    moving_image : np.ndarray with shape (..., M, N)
        The moving image. This image is shifted and the resulting phase correlation is measured.
    eps : float, optional
        Small value to avoid division by zero. Default is 1e-8.

    Returns
    -------
    np.ndarray
        The cross-power spectrum, with the same shape as the input images.
    """
    if not _broadcastable(static_image, moving_image):
        raise ValueError("Images must be broadcastable.")

    if static_image.shape[-2:] != moving_image.shape[-2:]:
        raise ValueError("Images must have the same shape in the last two dimensions.")

    # measure cross-power-spectrum
    fft_static = np.fft.fft2(static_image, axes=(-2, -1))
    fft_moving_conj = np.conj(np.fft.fft2(moving_image, axes=(-2, -1)))
    R = fft_static * fft_moving_conj
    R /= eps + np.abs(R)
    return R


def phase_correlation(static_image: np.ndarray, moving_image: np.ndarray, eps: float = 1e-8):
    """Measure the phase correlation between two images.

    Computes the phase correlation in the fourier domain with the following formula:
    .. math::
        R = F^{-1}(F(static\_image) * conj(F(moving\_image)))

    where F is the fourier transform and conj is the complex conjugate.

    Returns the real part, which describes the phase correlation of the two images
    after shifting the moving image.

    Note: broadcasting between the two images is accepted as long as the last two
    dimensions have equal shapes, which represent the "image" dimensions.

    Parameters
    ----------
    static_image : np.ndarray with shape (..., M, N)
        The static image.
    moving_image : np.ndarray with shape (..., M, N)
        The moving image. This image is shifted and the resulting phase correlation is measured.
    eps : float, optional
        Small value to avoid division by zero. Default is 1e-8.

    Returns
    -------
    np.ndarray
        The phase correlation map, representing the correlation for every shift
        of moving_image relative to static_image. The shape will be equal to the
        (broadcasted) shape of the input images.
    """
    R = cross_power_spectrum(static_image, moving_image, eps=eps)
    return np.fft.ifft2(R).real
