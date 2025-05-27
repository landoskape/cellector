from typing import List, Tuple, Optional, Sequence, Callable, Union
import warnings
import functools
import numpy as np
import numba as nb
from scipy.ndimage import binary_dilation, generate_binary_structure


def broadcastable(x: np.ndarray, y: np.ndarray) -> bool:
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


def deprecated(reason: str, version: Optional[str] = None):
    """
    Decorator to mark functions as deprecated.

    Parameters
    ----------
    reason : str
        Reason why the function is being deprecated and what to use instead.
    version : str, optional
        Version the function will be removed in.
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warning_msg = f"{func.__name__} is deprecated.\n{reason}"
            if version:
                warning_msg += f"\nIt will be removed in version {version}."
            warnings.warn(warning_msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


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


def cat_planes(
    list_of_sequences: List[Union[Sequence, np.ndarray]],
) -> List[Union[Sequence, np.ndarray]]:
    """Concatenate a sequence of sequences into a single sequence.

    Parameters
    ----------
    list_of_sequences : List[Union[Sequence, np.ndarray]]
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
    of each plane independently, use the split_planes function. Note that
    cat_planes and split_planes are not inverse operations unless the ROIs
    start in order of planes.
    """
    if all(isinstance(seq, np.ndarray) for seq in list_of_sequences):
        return np.concatenate(list_of_sequences)
    return [item for subsequence in list_of_sequences for item in subsequence]


def split_planes(array: np.ndarray, plane_idx: np.ndarray) -> List[np.ndarray]:
    """
    Split a numpy array into a list of numpy arrays based on the plane index.

    Parameters
    ----------
    array : np.ndarray
        A single numpy array to split into multiple numpy arrays.
    plane_idx : np.ndarray
        An array of integers indicating the plane index for each element in the sequence.

    Returns
    -------
    list
        A list of numpy arrays, each containing the elements for a single plane.

    Notes
    -----
    This function is used to split concatenated ROI data back into separate
    lists for each plane after operations have been performed on all ROIs at once
    using the cat_planes function. It is not the inverse of cat_planes, because
    ROIs need not be contiguous but cat_planes will just concatenate them!
    """
    max_plane = np.max(plane_idx)
    array_by_plane = []
    for plane in range(max_plane + 1):
        idx_to_plane = plane_idx == plane
        array_by_plane.append(array[idx_to_plane])
    return array_by_plane


def flatten_roi_data(
    lam: List[np.ndarray],
    ypix: List[np.ndarray],
    xpix: List[np.ndarray],
    zpix: Optional[List[np.ndarray]] = None,
) -> dict[str, np.ndarray]:
    """Get flattened ROI data for use in parallel processing.

    Returns flattened arrays for intensity values, y-coordinates, x-coordinates in which
    the data is concatenated across ROIs such that each is a single numpy array. The ROI
    index is also returned as a single numpy array with the same length as the other arrays
    to indicate which ROI each value belongs to.

    Parameters
    ----------
    lam : list of np.ndarrays
        Intensity values for each pixel in each ROI.
    zpix : list of np.ndarrays | np.ndarray
        Z-coordinates for each ROI. If 2D data, this should be a single np.ndarray of the z-plane.
        If volumetric data, this should be a list of np.ndarrays, one for each ROI.
    ypix : list of np.ndarrays
        Y-coordinates of pixels in each ROI.
    xpix : list of np.ndarrays
        X-coordinates of pixels in each ROI.

    Returns
    -------
    dict
        A dictionary containing the flattened arrays for intensity values, z-coordinates, y-coordinates,
        x-coordinates, and ROI index associated with each lam/z/y/x value. If zpix is not provided, then
        it will not be included in the output dictionary.
    """
    lam_flat = np.concatenate(lam)
    ypix_flat = np.concatenate(ypix)
    xpix_flat = np.concatenate(xpix)
    roi_idx = np.repeat(np.arange(len(lam)), [len(arr) for arr in lam])
    output = dict(
        lam_flat=lam_flat,
        ypix_flat=ypix_flat,
        xpix_flat=xpix_flat,
        roi_idx=roi_idx,
    )
    if zpix is not None:
        zpix_flat = np.concatenate(zpix)
        output["zpix_flat"] = zpix_flat
    return output


def get_roi_centroid(
    lam: np.ndarray,
    zpix: np.ndarray | int,
    ypix: np.ndarray,
    xpix: np.ndarray,
    method: str = "weightedmean",
    asint: bool = True,
) -> Tuple[int, int]:
    """Calculate the centroid coordinates of an ROI.

    Parameters
    ----------
    lam : np.ndarray
        Intensity values for each pixel in the ROI.
    zpix : np.ndarray | int
        Z-coordinates of pixels in the ROI for volumetric data, or just
        the z-plane for 2D data.
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
    zc, yc, xc : tuple of int or float
        Z, Y, and X coordinates of the centroid. Returns as integers by default,
        but will return floats if asint=False and method="weightedmean".
    """
    if method == "weightedmean":
        if isinstance(zpix, np.ndarray):
            zc = np.sum(lam * zpix) / np.sum(lam)
        else:
            zc = zpix
        yc = np.sum(lam * ypix) / np.sum(lam)
        xc = np.sum(lam * xpix) / np.sum(lam)
        if asint:
            zc, yc, xc = int(zc), int(yc), int(xc)
    elif method == "median":
        if isinstance(zpix, np.ndarray):
            zc = int(np.median(zpix))
        else:
            zc = int(zpix)
        yc = int(np.median(ypix))
        xc = int(np.median(xpix))
    else:
        raise ValueError(
            f"Invalid method ({method}). Must be 'weightedmean' or 'median'"
        )
    return zc, yc, xc


def get_roi_centroids(
    lam: List[np.ndarray],
    zpix: np.ndarray | List[np.ndarray],
    ypix: List[np.ndarray],
    xpix: List[np.ndarray],
    method: str = "weightedmean",
    asint: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the centroid of each ROI in a list of ROIs.

    Parameters
    ----------
    lam : list of np.ndarrays
        Intensity values for each pixel in each ROI.
    zpix : np.ndarray | list of np.ndarrays
        Z-coordinates of each ROI -- if volumetric data this is a list of np.ndarrays
        for each ROI, if not volumetric this is a single np.ndarray of the z-plane for
        each ROI.
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
    zc, yc, xc : tuple of np.ndarrays of int or float
        Z, Y, and X coordinates of each ROI centroid. Returns as integers by default,
        but will return floats if asint=False and method="weightedmean".

    See Also
    --------
    get_roi_centroid : The underlying function for centroid calculation.
    """
    centroids = [
        get_roi_centroid(l, z, y, x, method=method, asint=asint)
        for (l, z, y, x) in zip(lam, zpix, ypix, xpix)
    ]
    zc, yc, xc = transpose(centroids)
    return zc, yc, xc


def get_mask_volume(
    flattened_roi_data: dict[str, np.ndarray],
    num_rois: int,
    shape: Tuple[int, int],
    volumetric: bool = False,
) -> np.ndarray:
    """Create a mask stack using Numba for parallel processing.

    For optimization with numba, the input is flattened arrays of pixel coordinates and intensities
    concatenated across ROIs, with the ROI index indicated by ``roi_idx``. The volume is created by
    assigning the intensity values to the appropriate pixel coordinates in the mask stack with zeros
    everywhere else. Each slice of the mask stack corresponds to a single ROI.

    Parameters
    ----------
    flattened_roi_data : dict[str, np.ndarray]
        lam_flat : np.ndarray
            Intensity values for each pixel in each ROI.
            In format (num_pixels,) with values from each ROI concatenated with each other.
        zpix : np.ndarray
            Z-coordinates of pixels in each ROI.
            Same format as lam.
        ypix_flat : np.ndarray
            Y-coordinates of pixels in each ROI.
            Same format as lam.
        xpix_flat : np.ndarray
            X-coordinates of pixels in each ROI.
            Same format as lam.
        roi_idx : np.ndarray
            ROI index for each value in lam, ypix, xpix
    num_rois : int
        Number of ROIs described in the input arrays.
    shape : tuple of int
        Shape of the mask stack to create (height and width).
    volumetric : bool, optional
        Whether the data is volumetric. If True, will include all planes. Default is False.

    Returns
    -------
    np.ndarray
        A stack of masks images for each ROI.
        If volumetric=False, has shape (num_rois, height, width).
        If volumetric=True, has shape (num_rois, num_planes, height, width).
    """
    mask_stack = np.zeros((num_rois, *shape), dtype=np.float32)
    if volumetric:
        return _nb_get_mask_volume_volumetric(
            flattened_roi_data["lam_flat"],
            flattened_roi_data["zpix_flat"],
            flattened_roi_data["ypix_flat"],
            flattened_roi_data["xpix_flat"],
            flattened_roi_data["roi_idx"],
            mask_stack,
        )
    else:
        return _nb_get_mask_volume(
            flattened_roi_data["lam_flat"],
            flattened_roi_data["ypix_flat"],
            flattened_roi_data["xpix_flat"],
            flattened_roi_data["roi_idx"],
            mask_stack,
        )


@nb.njit(parallel=True)
def _nb_get_mask_volume(
    lam: np.ndarray,
    ypix: np.ndarray,
    xpix: np.ndarray,
    roi_idx: np.ndarray,
    mask_stack: np.ndarray,
) -> np.ndarray:
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
    flattened_roi_data: dict[str, np.ndarray],
    centroids: dict[str, np.ndarray],
    width: Optional[int] = 15,
    fill_value: Optional[float] = 0.0,
    num_planes: Optional[int] = None,
    volumetric: bool = False,
) -> np.ndarray:
    """Create a stack of centered masks for each ROI.

    For optimization with numba, the input is flattened arrays of pixel coordinates and intensities
    concatenated across ROIs, with the ROI index indicated by ``roi_idx``. The centroids are provided
    as a tuple of y and x coordinates for each ROI. The mask stack is created by centering each mask
    around the centroid of the ROI, with a width of ``width`` pixels on each side. Any pixels outside
    the mask footprint are filled with the value specified by ``fill_value``.

    Parameters
    ----------
    flattened_roi_data : dict[str, np.ndarray]
        lam : np.ndarray
            Intensity values for each pixel in each ROI.
            In format (num_pixels,) with values from each ROI concatenated with each other.
        zpix : np.ndarray
            Z-coordinates of pixels in each ROI.
            Same format as lam.
        ypix : np.ndarray
            Y-coordinates of pixels in each ROI.
            Same format as lam.
        xpix : np.ndarray
            X-coordinates of pixels in each ROI.
            Same format as lam.
        roi_idx : np.ndarray
            ROI index for each value in lam, ypix, xpix, zpix
    centroids : dict[str, np.ndarray]
        Dictionary of z, y, and x centroids for each ROI.
    width : int, optional
        Width in pixels around the ROI centroid. Default is 15.
    fill_value : float, optional
        Value to use as the background. Default is 0.0.
    num_planes : int, optional
        Number of planes in the data. Required if ``volumetric`` is True.
    volumetric : bool, optional
        Whether the data is volumetric. If True, will include all planes. Default is False.

    Returns
    -------
    np.ndarray
        A stack of centered masks for each ROI.
        If volumetric=False, has shape (num_rois, 2 * width + 1, 2 * width + 1).
        If volumetric=True, has shape (num_rois, num_planes, 2 * width + 1, 2 * width + 1).
    """
    if volumetric and num_planes is None:
        raise ValueError("num_planes must be provided if volumetric is True")
    if volumetric:
        mask_stack = _nb_get_centered_masks_volumetric(
            flattened_roi_data["lam_flat"],
            flattened_roi_data["zpix_flat"],
            flattened_roi_data["ypix_flat"],
            flattened_roi_data["xpix_flat"],
            flattened_roi_data["roi_idx"],
            centroids["yc"],
            centroids["xc"],
            width,
            fill_value,
            num_planes,
        )
    else:
        mask_stack = _nb_get_centered_masks(
            flattened_roi_data["lam_flat"],
            flattened_roi_data["ypix_flat"],
            flattened_roi_data["xpix_flat"],
            flattened_roi_data["roi_idx"],
            centroids["yc"],
            centroids["xc"],
            width,
            fill_value,
        )
    return mask_stack


@nb.njit(parallel=True)
def _nb_get_centered_masks(
    lam: np.ndarray,
    ypix: np.ndarray,
    xpix: np.ndarray,
    roi_idx: np.ndarray,
    yc: np.ndarray,
    xc: np.ndarray,
    width: int,
    fill_value: float,
) -> np.ndarray:
    """Create a stack of centered masks for each ROI using Numba for parallel processing."""
    mask_stack = np.full(
        (len(yc), 2 * width + 1, 2 * width + 1), fill_value, dtype=np.float32
    )
    for idx in nb.prange(len(lam)):
        c_yc, c_xc = yc[roi_idx[idx]], xc[roi_idx[idx]]
        cyidx = ypix[idx] - c_yc + width
        cxidx = xpix[idx] - c_xc + width
        if (
            cyidx >= 0
            and cyidx < 2 * width + 1
            and cxidx >= 0
            and cxidx < 2 * width + 1
        ):
            mask_stack[roi_idx[idx], cyidx, cxidx] = lam[idx]
    return mask_stack


@nb.njit(parallel=True)
def _nb_get_centered_masks_volumetric(
    lam: np.ndarray,
    zpix: np.ndarray,
    ypix: np.ndarray,
    xpix: np.ndarray,
    roi_idx: np.ndarray,
    yc: np.ndarray,
    xc: np.ndarray,
    width: int,
    fill_value: float,
    num_planes: int,
) -> np.ndarray:
    """Create a stack of centered masks volumes for each ROI using Numba for parallel processing."""
    mask_stack = np.full(
        (len(yc), num_planes, 2 * width + 1, 2 * width + 1),
        fill_value,
        dtype=np.float32,
    )
    for idx in nb.prange(len(lam)):
        c_yc, c_xc = yc[roi_idx[idx]], xc[roi_idx[idx]]
        cyidx = ypix[idx] - c_yc + width
        cxidx = xpix[idx] - c_xc + width
        if (
            cyidx >= 0
            and cyidx < 2 * width + 1
            and cxidx >= 0
            and cxidx < 2 * width + 1
        ):
            mask_stack[roi_idx[idx], zpix[idx], cyidx, cxidx] = lam[idx]
    return mask_stack


def _get_roi_bounds(
    center: int, width: int, image_dim: int
) -> Tuple[int, int, int, int]:
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


def get_centered_reference(
    reference: np.ndarray,
    centroids: dict[str, np.ndarray],
    width: Optional[int] = 15,
    fill_value: Optional[float] = 0.0,
    volumetric: bool = False,
) -> np.ndarray:
    """Create a stack of centered reference images on each ROI.

    Parameters
    ----------
    reference : np.ndarray
        Reference images for each plane (stack of 2D images).
    centroids : dict[str, np.ndarray]
        Dictionary of z, y, and x centroids for each ROI.
    width : int, optional
        Width in pixels around the ROI centroid. Default is 15.
    fill_value : float, optional
        Value to use as the background. Default is 0.0.
    volumetric : bool, optional
        Whether the data is volumetric. If True, will include all planes. Default is False.

    Returns
    -------
    np.ndarray
        A 3D stack of centered reference images for each ROI.
        If volumetric=False, has shape (num_rois, 2 * width + 1, 2 * width + 1).
        If volumetric=True, has shape (num_rois, num_planes, 2 * width + 1, 2 * width + 1).
    """
    if not isinstance(width, int) or width < 0:
        raise ValueError("Width must be a non-negative integer")

    # Preallocate reference stack array
    num_rois = len(centroids["yc"])
    num_planes = reference.shape[0]
    if volumetric:
        ref_stack = np.full(
            (num_rois, num_planes, 2 * width + 1, 2 * width + 1),
            fill_value,
            dtype=np.float32,
        )
    else:
        ref_stack = np.full(
            (num_rois, 2 * width + 1, 2 * width + 1), fill_value, dtype=np.float32
        )

    # For each ROI, fill in a patch of the appropriate reference image into the centered_stack array
    for idx in range(num_rois):
        ystart, yend, ystart_offset, yend_offset = _get_roi_bounds(
            centroids["yc"][idx], width, reference[centroids["zc"][idx]].shape[0]
        )
        xstart, xend, xstart_offset, xend_offset = _get_roi_bounds(
            centroids["xc"][idx], width, reference[centroids["zc"][idx]].shape[1]
        )
        y_centered_slice = slice(ystart_offset, 2 * width + 1 + yend_offset)
        x_centered_slice = slice(xstart_offset, 2 * width + 1 + xend_offset)
        if volumetric:
            ref_stack[idx, :, y_centered_slice, x_centered_slice] = reference[
                :, ystart:yend, xstart:xend
            ]
        else:
            ref_stack[idx, y_centered_slice, x_centered_slice] = reference[
                centroids["zc"][idx]
            ][ystart:yend, xstart:xend]
    return ref_stack


def center_surround(
    centered_masks: np.ndarray,
    iterations: Optional[int] = 7,
    volumetric: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the center and surround footprints of a stack of centered masks.

    The center footprint is the binary mask of the ROI itself, while the surround
    footprint is a binary mask of the surrounding region of the ROI. The surround
    region is defined as the dilation of the center region by a 3x3 cross structuring
    element for 7 iterations.

    Parameters
    ----------
    centered_masks : np.ndarray
        The centered mask images for each ROI. Should have shape (num_rois, height, width).
        If volumetric=True, should have shape (num_rois, num_planes, height, width).
    iterations : int, optional
        The number of iterations for the binary dilation. Default is 7.
    volumetric : bool, optional
        Whether the data is volumetric. If True, will include the z-axis
        in the computation. Default is False.

    Returns
    -------
    center, surround : Tuple[np.ndarray, np.ndarray]
        The center and surround binary masks for each ROI.
    """
    # Footprint of ROI defined as any non-zero pixel
    center = np.where(centered_masks > 0, 1, 0)

    # This structuring element allows parallelization across ROIs without interference
    if volumetric:
        stack_structure = np.stack(
            (
                np.zeros((3, 3, 3), dtype=bool),
                generate_binary_structure(3, 1),
                np.zeros((3, 3, 3), dtype=bool),
            )
        )
    else:
        stack_structure = np.stack(
            (
                np.zeros((3, 3), dtype=bool),
                generate_binary_structure(2, 1),
                np.zeros((3, 3), dtype=bool),
            )
        )
    surround = binary_dilation(center, structure=stack_structure, iterations=iterations)

    # Return as boolean arrays for use in indices and as masks
    return center.astype(bool), surround.astype(bool)


def surround_filter(
    centered_masks: np.ndarray,
    centered_references: np.ndarray,
    iterations: Optional[int] = 7,
    volumetric: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
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
    volumetric : bool, optional
        Whether the data is volumetric. If True, will include the z-axis
        in the computation. Default is False.

    Returns
    -------
    masks_surround, references_surround : Tuple[np.ndarray, np.ndarray]
        The centered mask and reference images for the surround region of each ROI.
        Anything outside the surround region will be filled with np.nan.
    """
    surround = center_surround(
        centered_masks,
        iterations=iterations,
        volumetric=volumetric,
    )[1]
    masks_surround = np.where(surround, centered_masks, np.nan)
    references_surround = np.where(surround, centered_references, np.nan)
    return masks_surround, references_surround


def cross_power_spectrum(
    static_image: np.ndarray,
    moving_image: np.ndarray,
    eps: float = 1e6,
    volumetric: bool = False,
) -> np.ndarray:
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
    volumetric : bool, optional
        Whether to compute the cross-power spectrum for volumetric data. If so, will include the z-axis
        in the computation. Default is False.

    Returns
    -------
    np.ndarray
        The cross-power spectrum, with the same shape as the input images.
    """
    if not broadcastable(static_image, moving_image):
        raise ValueError("Images must be broadcastable.")

    if static_image.shape[-2:] != moving_image.shape[-2:]:
        raise ValueError("Images must have the same shape in the last two dimensions.")

    if volumetric:
        axis = (-3, -2, -1)
        if static_image.shape[-3] != moving_image.shape[-3]:
            raise ValueError("Volumetric images must have the same number of planes.")
    else:
        axis = (-2, -1)

    # measure cross-power-spectrum
    fft_static = np.fft.fft2(static_image, axes=axis)
    fft_moving_conj = np.conj(np.fft.fft2(moving_image, axes=axis))
    R = fft_static * fft_moving_conj
    R /= eps + np.abs(R)
    return R


def phase_correlation(
    static_image: np.ndarray,
    moving_image: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
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


def phase_correlation_zero(
    centered_masks: np.ndarray,
    centered_reference: np.ndarray,
    eps: float = 1e6,
    volumetric: bool = False,
) -> np.ndarray:
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
    volumetric : bool, optional
        Whether to compute the phase correlation for volumetric data. If so, will include the z-axis
        in the computation. Default is False.

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
    R = cross_power_spectrum(
        centered_masks,
        centered_reference,
        eps=eps,
        volumetric=volumetric,
    )
    if volumetric:
        center_value = np.sum(R, axis=(-3, -2, -1)) / np.prod(R.shape[-3:])
    else:
        center_value = np.sum(R, axis=(-2, -1)) / np.prod(R.shape[-2:])
    return center_value.real


def in_vs_out(
    centered_masks: np.ndarray,
    centered_reference: np.ndarray,
    iterations: int = 7,
    volumetric: bool = False,
) -> np.ndarray:
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
    centered_masks : np.ndarray with shape (..., (Z), M, N)
        The centered mask images for each ROI.
    centered_reference : np.ndarray with shape (..., (Z), M, N)
        The centered reference image (centered on each ROI).
    iterations : int, optional
        The number of iterations to dilate the mask for the surround. Default is 7.
    volumetric : bool, optional
        Whether to compute the in vs. out feature for volumetric data. If so, will include the z-axis
        in the computation. Default is False.

    Returns
    -------
    np.ndarray
        The ratio of the intensity inside the mask to the intensity inside plus outside the mask.
        Shape will be (N,), where N is the number of ROIs. If the total sum of the inside and
        outside intensities is 0, then the value will return as 0.
    """
    if volumetric:
        axis = (-3, -2, -1)
    else:
        axis = (-2, -1)
    center, surround = center_surround(
        centered_masks,
        iterations=iterations,
        volumetric=volumetric,
    )
    inside_sum = np.sum(center * centered_reference, axis=axis)
    outside_sum = np.sum(surround * centered_reference, axis=axis)
    total_sum = inside_sum + outside_sum
    total_sum[total_sum == 0] = np.inf
    return inside_sum / total_sum


def dot_product(
    lam: List[np.ndarray],
    ypix: List[np.ndarray],
    xpix: List[np.ndarray],
    zpix: Union[np.ndarray, List[np.ndarray]],
    reference: np.ndarray,
    volumetric: bool = False,
) -> np.ndarray:
    """Measure the normalized dot product between the masks and the reference images.

    Will take the inner product of the mask weights with the reference image, and divide
    by the norm of the mask weights. This is a measure of how well the mask points toward
    intensity features in the reference image.

    Uses lam, xpix, ypix (and zpix if volumetric=True) instead of mask images for efficiency.

    Parameters
    ----------
    lam : List[np.ndarray]
        List of mask weights for each ROI.
    ypix : List[np.ndarray]
        List of y-pixel indices for each ROI.
    xpix : List[np.ndarray]
        List of x-pixel indices for each ROI.
    zpix : Union[np.ndarray, List[np.ndarray]]
        List of plane indices for each ROI, unless volumetric=True, in which case it is a list
        of z-plane indices for each ROI.
    reference : np.ndarray
        The reference images.
    volumetric : bool, optional
        Whether to compute the dot product for volumetric data. If so, will include the z-axis
        in the computation. Default is False.

    Returns
    -------
    np.ndarray
        The dot product between the masks and the reference images. Shape will be (N,),
        where N is the number of ROIs.
    """
    dot_product = np.zeros(len(lam))
    for idx, (l, y, x, p) in enumerate(zip(lam, ypix, xpix, zpix)):
        if volumetric:
            dot = np.sum(reference[p, y, x] @ l / np.linalg.norm(l))
        else:
            ref = reference[p]
            dot = np.sum(ref[y, x] @ l / np.linalg.norm(l))
        dot_product[idx] = dot
    return dot_product


def compute_correlation(
    centered_masks: np.ndarray,
    centered_references: np.ndarray,
    volumetric: bool = False,
) -> np.ndarray:
    """Measure the correlation coefficient between the masks and the reference images.

    The masks and reference images should have the same shape and be centered on each other.
    It is expected (but not required) that the masks and references are filtered to have
    np.nan outside the surround of each ROI to focus the correlation on the local structure
    near the ROI rather than the full square provided by the centering stacks. Since the
    surround is larger than the ROI by definition, this is effectively comparing the local
    data in the ROI with the zeros outside the ROI.

    Parameters
    ----------
    centered_masks : np.ndarray with shape (..., (Z), M, N)
        The centered mask images for each ROI.
    centered_reference : np.ndarray with shape (..., (Z), M, N)
        The centered reference image (centered on each ROI).
    volumetric : bool, optional
        Whether the data is volumetric. If True, will include the z-axis
        in the computation. Default is False.

    Returns
    -------
    np.ndarray
        The correlation coefficient between the masks and the reference images. Shape will be (N,),
        where N is the number of ROIs.

    See Also
    --------
    surround_filter : Function to filter the masks and references to have np.nan outside the ROI.
    """
    if volumetric:
        axis = (-3, -2, -1)
    else:
        axis = (-2, -1)
    u_ref = np.nanmean(centered_references, axis=axis, keepdims=True)
    u_mask = np.nanmean(centered_masks, axis=axis, keepdims=True)
    s_ref = np.nanstd(centered_references, axis=axis)
    s_mask = np.nanstd(centered_masks, axis=axis)
    N = np.sum(~np.isnan(centered_masks), axis=axis)
    return (
        np.nansum((centered_references - u_ref) * (centered_masks - u_mask), axis=axis)
        / N
        / s_ref
        / s_mask
    )
