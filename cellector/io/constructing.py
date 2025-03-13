from typing import List, Union, Tuple
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
from ..utils import transpose, cat_planes
from ..roi_processor import RoiProcessor
from .operations import clear_cellector_files


# ---------------------------------------------------------------------------------------
# Methods for creating RoiProcessor objects from suite2p directories
# ---------------------------------------------------------------------------------------
def _get_s2p_folders(suite2p_dir: Union[Path, str]) -> Tuple[List[Path], bool]:
    """Get list of directories for each plane in a suite2p directory.

    Parameters
    ----------
    suite2p_dir : Path or str
        Path to the suite2p directory, which contains directories for each plane in the
        format "plane0", "plane1", etc. If the suite2p directory doesn't contain plane*
        folders but contains stat.npy and ops.npy files, it will be treated as a single
        plane. Otherwise, an error will be raised.

    Returns
    -------
    plane_folders : list of Paths
        List of directories for each plane in the suite2p directory.
    """
    suite2p_dir = Path(suite2p_dir)
    planes = suite2p_dir.glob("plane*")
    if planes:
        plane_folders = list(planes)

        # Make sure all relevant files are present
        if not all(folder.is_dir() for folder in plane_folders):
            raise ValueError(f"Plane paths are not all directories in {suite2p_dir}!")
        if not all((folder / "stat.npy").exists() for folder in plane_folders):
            raise FileNotFoundError(
                f"Could not find stat.npy files in each folder {suite2p_dir}!"
            )
        if not all((folder / "ops.npy").exists() for folder in plane_folders):
            raise FileNotFoundError(
                f"Could not find any ops.npy files in {suite2p_dir}!"
            )

    # If stat.npy and ops.py are in the s2p_dir itself, assume it's a single plane without a plane folder
    elif (suite2p_dir / "stat.npy").exists() and (suite2p_dir / "ops.npy").exists():
        plane_folders = [suite2p_dir]

    else:
        raise FileNotFoundError(
            f"Could not find any plane directories or stat.npy / ops.npy files in {suite2p_dir}!"
        )

    return plane_folders


def _get_s2p_data(
    s2p_folders: List[Path], reference_key: str = "meanImg_chan2"
) -> Tuple[list[list[dict]], list[np.ndarray]]:
    """Get list of stats and chan2 reference images from all planes in a suite2p directory.

    suite2p saves the statistics and reference images for each plane in separate
    directories. This function reads the statistics and reference images for each plane
    and returns them as lists.

    Parameters
    ----------
    s2p_folders : list of Path
        List of directories that contain the suite2p output for each plane (stat.npy and ops.npy).
    reference_key : str, optional
        Key to use for the reference image. Default is "meanImg_chan2".

    Returns
    -------
    stats : list of list of dictionaries
        Each element of stats is a list of dictionaries containing ROI statistics for each plane.
    references : list of np.ndarrays
        Each element of references is an image (usually of average red fluorescence) for each plane.
    """
    stats = []
    references = []
    for folder in s2p_folders:
        stats.append(np.load(folder / "stat.npy", allow_pickle=True))
        ops = np.load(folder / "ops.npy", allow_pickle=True).item()
        if reference_key not in ops:
            raise ValueError(
                f"Reference key ({reference_key}) not found in ops.npy file ({folder / 'ops.npy'})!"
            )
        references.append(ops[reference_key])
    if not all(ref.shape == references[0].shape for ref in references):
        raise ValueError("Reference images must have the same shape as each other!")
    if not all(ref.ndim == 2 for ref in references):
        raise ValueError("Reference images must be 2D arrays!")
    return stats, references


def _get_s2p_redcell(s2p_folders: List[Path]) -> list[np.ndarray]:
    """Get red cell probability masks from all planes in a suite2p directory.

    Extracts the red cell probability masks from each plane in a suite2p directory
    and returns them as a list of numpy arrays. The red cell probability masks are
    saved in the "redcell.npy" file in each plane directory in which the first column
    is a red cell assigment and the second column is the probability of each ROI being
    a red cell.

    Parameters
    ----------
    s2p_folders : list of Path
        List of directories that contain the suite2p output for each plane (redcell.npy).

    Returns
    -------
    redcell : list of np.ndarrays
        List of red cell probabilities for each plane. Each array has length N corresponding
        to the number of ROIs in that plane.
    """
    redcell = []
    for folder in s2p_folders:
        if not (folder / "redcell.npy").exists():
            raise FileNotFoundError(f"Could not find redcell.npy file in {folder}!")
        c_redcell = np.load(folder / "redcell.npy")
        redcell.append(c_redcell[:, 1])
    return redcell


def create_from_suite2p(
    suite2p_dir: Union[Path, str],
    use_redcell: bool = True,
    reference_key: str = "meanImg_chan2",
    extra_features: dict = {},
    autocompute: bool = True,
    clear_existing: bool = False,
    save_features: bool = True,
) -> RoiProcessor:
    """Create a RoiProcessor object from a suite2p directory.

    Parameters
    ----------
    suite2p_dir : Path or str
        Path to the suite2p directory.
    use_redcell : bool, optional
        Whether to load redcell data from suite2p folders. Default is True.
    reference_key : str, optional
        Key to use for reference images in the suite2p folders. Default is "meanImg_chan2".
    extra_features : dict, optional
        Extra features to add to the RoiProcessor object. Default is empty.
    autocompute : bool, optional
        Whether to automatically compute all features for the RoiProcessor object. Default is True.
    clear_existing : bool, optional
        Whether to clear existing cellector files in the root directory. Default is False.
    save_features : bool, optional
        Whether to save the features to disk. As soon as features are computed, they will be saved
        in the cellector folder. Default is True.

    Returns
    -------
    roi_processor : RoiProcessor
        RoiProcessor object with suite2p masks and reference images loaded that uses the suite2p_dir as the root directory.
    """
    if clear_existing:
        clear_cellector_files(suite2p_dir)

    suite2p_dir = Path(suite2p_dir)
    suite2p_folders = _get_s2p_folders(suite2p_dir)
    num_planes = len(suite2p_folders)
    stats, references = _get_s2p_data(suite2p_folders, reference_key=reference_key)
    rois_per_plane = [len(stat) for stat in stats]

    if use_redcell:
        # Raises an error if it doesn't exist
        s2p_redcell = _get_s2p_redcell(suite2p_folders)
        extra_features["red_s2p"] = s2p_redcell

    # Build appropriate format for RoiProcessor (ROI data concatenated across planes)
    stats = cat_planes(stats)
    references = np.stack(references)
    plane_idx = np.repeat(np.arange(num_planes), rois_per_plane)

    # Initialize roi_processor object with suite2p data
    return RoiProcessor(
        suite2p_dir,
        stats,
        references,
        plane_idx,
        extra_features=extra_features,
        autocompute=autocompute,
        save_features=save_features,
    )


# ---------------------------------------------------------------------------------------
# Methods for creating RoiProcessor objects from mask volumes or pixel data directly
# ---------------------------------------------------------------------------------------
def _get_pixel_data_single(
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _get_pixel_data(mask_volume: np.ndarray, verbose: bool = True) -> list[dict]:
    """Get pixel data from a mask volume.

    Extracts the intensity values, y-coordinates, and x-coordinates from a mask volume
    where each slice of the volume corresponds to a single ROI.

    Parameters
    ----------
    mask_volume : np.ndarray
        A 3D mask volume with shape (num_rois, height, width) where each slice is a mask
        footprint with intensity values.
    verbose : bool, optional
        Whether to use a tqdm progress bar to show progress. Default is True.

    Returns
    -------
    stats : list of dict
        List of dictionaries with keys "lam", "ypix", and "xpix" for each ROI.
    """
    n_workers = max(2, cpu_count() - 2)
    try:
        with Pool(n_workers) as pool:
            iterable = (
                tqdm(mask_volume, desc="Extracting mask data", leave=False)
                if verbose
                else mask_volume
            )
            results = list(pool.imap(_get_pixel_data_single, iterable))

    except Exception as e:
        if "pool" in locals():
            pool.terminate()
            pool.join()
        raise e from None

    lam, ypix, xpix = transpose(results)
    stats = [dict(lam=l, ypix=y, xpix=x) for l, y, x in zip(lam, ypix, xpix)]
    return stats


def create_from_mask_volume(
    root_dir: Union[Path, str],
    mask_volume: np.ndarray,
    references: np.ndarray,
    plane_idx: np.ndarray,
    extra_features: dict = {},
    autocompute: bool = True,
    clear_existing: bool = False,
    save_features: bool = True,
) -> RoiProcessor:
    """Create a RoiProcessor object

    Parameters
    ----------
    root_dir : Path or str
        Path to the root directory which will be used for saving results.
    mask_volume : np.ndarray
        A 3D mask volume with shape (num_rois, height, width) where each slice is a mask
        footprint with intensity values and zeros elsewhere.
    references : np.ndarray
        A 3D reference volume with shape (num_planes, height, width) where each slice is a reference
        image for a plane containing the fluorescence values to compare masks to.
    plane_idx : np.ndarray
        A 1D array of plane indices for each ROI in the mask volume.
    extra_features : dict, optional
        Extra features to add to the RoiProcessor object. Default is empty.
    autocompute : bool, optional
        Whether to automatically compute all features for the RoiProcessor object. Default is True.
    clear_existing : bool, optional
        Whether to clear existing cellector files in the root directory. Default is False.
    save_features : bool, optional
        Whether to save the features to disk. As soon as features are computed, they will be saved
        in the cellector folder. Default is True.

    Returns
    -------
    roi_processor : RoiProcessor
        RoiProcessor object with roi masks and reference data loaded.
    """
    if clear_existing:
        clear_cellector_files(root_dir)
    stats = _get_pixel_data(mask_volume)
    return RoiProcessor(
        root_dir,
        stats,
        references,
        plane_idx,
        extra_features=extra_features,
        autocompute=autocompute,
        save_features=save_features,
    )


def create_from_pixel_data(
    root_dir: Union[Path, str],
    stats: List[dict],
    references: np.ndarray,
    plane_idx: np.ndarray,
    extra_features: dict = {},
    autocompute: bool = True,
    clear_existing: bool = False,
    save_features: bool = True,
) -> RoiProcessor:
    """Create a RoiProcessor object

    Parameters
    ----------
    root_dir : Path or str
        Path to the root directory which will be used for saving results.
    stats : List[dict]
        List of dictionaries with keys "lam", "ypix", and "xpix" for each ROI containing the intensity values,
        y-coordinates, and x-coordinates for each ROI.
    references : np.ndarray
        A 3D reference volume with shape (num_planes, height, width) where each slice is a reference
        image for a plane containing the fluorescence values to compare masks to.
    plane_idx : np.ndarray
        A 1D array of plane indices for each ROI in the mask volume.
    extra_features : dict, optional
        Extra features to add to the RoiProcessor object. Default is empty.
    autocompute : bool, optional
        Whether to automatically compute all features for the RoiProcessor object. Default is True.
    clear_existing : bool, optional
        Whether to clear existing cellector files in the root directory. Default is False.
    save_features : bool, optional
        Whether to save the features to disk. As soon as features are computed, they will be saved
        in the cellector folder. Default is True.

    Returns
    -------
    roi_processor : RoiProcessor
        RoiProcessor object with roi masks and reference data loaded.
    """
    if clear_existing:
        clear_cellector_files(root_dir)
    return RoiProcessor(
        root_dir,
        stats,
        references,
        plane_idx,
        extra_features=extra_features,
        autocompute=autocompute,
        save_features=save_features,
    )
