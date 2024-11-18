from typing import Union, Dict
from copy import copy
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
from .utils import transpose, get_s2p_data, get_s2p_redcell, split_planes
from .roi_processor import RoiProcessor


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


def get_pixel_data(mask_volume, verbose: bool = True):
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
    iterable = tqdm(mask_volume, desc="Extracting mask data", leave=False) if verbose else mask_volume
    with Pool(max(2, cpu_count() - 2)) as pool:
        results = pool.map(_get_pixel_data_single, iterable)
    lam, ypix, xpix = transpose(results)
    return [dict(lam=l, ypix=y, xpix=x) for l, y, x in zip(lam, ypix, xpix)]


class Suite2pLoader:
    def __init__(self, s2p_dir: Union[Path, str], reference_key: str = "meanImg_chan2", use_redcell: bool = True):

        self.s2p_dir = Path(s2p_dir)
        self.reference_key = reference_key

        # Get s2p folders, roi and reference data, and redcell data if it exists
        self.get_s2p_folders()
        self.num_planes = len(self.folders)
        self.stats, self.references = get_s2p_data(self.folders, reference_key=self.reference_key)

        # Get redcell data if it exists
        if use_redcell and all((folder / "redcell.npy").exists() for folder in self.folders):
            self.redcell = get_s2p_redcell(self.folders)
        else:
            raise FileNotFoundError(f"Could not find redcell.npy files in all folders {self.s2p_dir}!")

    def get_s2p_folders(self):
        """Get list of directories for each plane in a suite2p directory.

        Parameters
        ----------
        s2p_dir : Path
            Path to the suite2p directory, which contains directories for each plane in the
            format "plane0", "plane1", etc.

        Returns
        -------
        planes : list of Path
            List of directories for each plane in the suite2p directory.
        has_planes : bool
            Whether the suite2p directory contains directories for each plane.
        """
        planes = self.s2p_dir.glob("plane*")
        if planes:
            self.has_planes = True
            self.folders = list(planes)

            # Make sure all relevant files are present
            if not all(folder.is_dir() for folder in self.folders):
                raise FileNotFoundError(f"Could not find all plane directories in {self.s2p_dir}!")
            if not all((folder / "stat.npy").exists() for folder in self.folders):
                raise FileNotFoundError(f"Could not find stat.npy files in each folder {self.s2p_dir}!")
            if not all((folder / "ops.npy").exists() for folder in self.folders):
                raise FileNotFoundError(f"Could not find any ops.npy files in {self.s2p_dir}!")

        # If stat.npy and ops.py are in the s2p_dir itself, assume it's a single plane without a plane folder
        elif (self.s2p_dir / "stat.npy").exists() and (self.s2p_dir / "ops.npy").exists():
            self.has_planes = False
            self.folders = [self.s2p_dir]

        else:
            raise FileNotFoundError(f"Could not find any plane directories or stat.npy / ops.npy files in {self.s2p_dir}!")


def create_from_suite2p(
    suite2p_dir: Union[Path, str],
    use_redcell: bool = True,
    extra_features: dict = {},
    autocompute: bool = True,
    clear_existing: bool = False,
):
    """Create a RoiProcessor object from a suite2p directory.

    Parameters
    ----------
    suite2p_dir : Path or str
        Path to the suite2p directory.
    use_redcell : bool, optional
        Whether to attempt to load redcell data from suite2p folders. Default is True.
    extra_features : dict, optional
        Extra features to add to the RoiProcessor object. Default is empty.
    autocompute : bool, optional
        Whether to automatically compute all features for the RoiProcessor object. Default is True.

    Returns
    -------
    roi_processor : RoiProcessor
        RoiProcessor object with suite2p data loaded.
    """
    s2p_data = Suite2pLoader(suite2p_dir, use_redcell=use_redcell)
    if s2p_data.redcell is not None:
        extra_features["red_s2p"] = s2p_data.redcell

    if clear_existing:
        for folder in s2p_data.folders:
            save_dir = folder / "cellector"
            if save_dir.exists():
                for file in save_dir.glob("*"):
                    file.unlink()
            # remove save_dir
            save_dir.rmdir()

    # Initialize roi_processor object with suite2p data
    return RoiProcessor(s2p_data.stats, s2p_data.references, s2p_data.folders, extra_features=extra_features, autocompute=autocompute)


def save_selection(
    roi_processor: RoiProcessor,
    idx_selection: np.ndarray,
    criteria: Dict[str, list],
    use_criteria: Dict[str, list],
    manual_selection: np.ndarray,
):
    """Save features, feature criteria, and manual labels to target folders.

    Saves the selection indices, feature values, feature criteria, and manual selection labels to the target folders
    defined in the roi_processor object. The feature values are saved as .npy files with the feature name as the filename.
    The feature criteria are saved as .npy files with the feature name and "_criteria" appended to the filename.
    The manual selection labels are saved as "manual_selection.npy".

    Parameters
    ----------
    roi_processor : RoiProcessor
        RoiProcessor object with features and folders to save to.
    idx_selection : np.ndarray
        Selection indices for each ROI. Should be a numpy array with shape (num_rois,) where each value is a boolean indicating
        whether the ROI is selected my meeting all feature criteria.
    criteria : Dict[str, list]
        Dictionary of feature criteria for each feature. Each value in the dictionary should be a 2 element list containing
        the minimum and maximum values for the feature.
    use_criteria: Dict[str, list]
        Dictionary of whether to use the feature criteria for each feature. Each value in the dictionary should be a 2 element
        list containing booleans indicating whether to use the minimum and maximum values for the feature.
    manual_selection : np.ndarray
        Manual selection labels for each ROI. Shape should be (num_rois, 2), where the first column is the manual label
        and the second column is whether or not to use a manual label for that cell.
    """
    # Check that everything has the expected shapes
    if idx_selection.shape[0] != roi_processor.num_rois:
        raise ValueError(f"Selection indices have shape {idx_selection.shape} but expected {roi_processor.num_rois}!")
    if (manual_selection.shape[0] != roi_processor.num_rois) or (manual_selection.shape[1] != 2):
        raise ValueError(f"Manual selection labels have shape {manual_selection.shape} but expected ({roi_processor.num_rois}, 2)!")
    for name, value in criteria.items():
        if name not in roi_processor.features:
            raise ValueError(f"Feature {name} not found in roi_processor features!")
        if len(value) != 2:
            raise ValueError(f"Feature criteria {name} has shape {value.shape} but expected (2,)!")
    if any(feature not in criteria for feature in roi_processor.features):
        raise ValueError(f"Feature criteria missing for features: {set(roi_processor.features) - set(criteria)}!")

    # Make folders for each plane
    for folder in roi_processor.save_folders:
        save_dir = folder / "cellector"
        save_dir.mkdir(exist_ok=True)

    # Save features values for each plane
    features_by_plane = {name: split_planes(value, roi_processor.rois_per_plane) for name, value in roi_processor.features.items()}
    manual_selection_by_plane = split_planes(manual_selection, roi_processor.rois_per_plane)
    for name, values in features_by_plane.items():
        for iplane, folder in enumerate(roi_processor.save_folders):
            np.save(folder / "cellector" / f"{name}.npy", values[iplane])
    for iplane, folder in enumerate(roi_processor.save_folders):
        np.save(folder / "cellector" / "manual_selection.npy", manual_selection_by_plane[iplane])

    # Save selection indices
    idx_selection_by_plane = split_planes(idx_selection, roi_processor.rois_per_plane)
    for iplane, folder in enumerate(roi_processor.save_folders):
        np.save(folder / "cellector" / "selection.npy", idx_selection_by_plane[iplane])

    # Save feature criteria
    for name, value in criteria.items():
        print(name, value, type(value))
        save_criteria = copy(value)
        if not use_criteria[name][0]:
            save_criteria[0] = None
        if not use_criteria[name][1]:
            save_criteria[1] = None
        print(save_criteria, use_criteria[name])
        for iplane, folder in enumerate(roi_processor.save_folders):
            np.save(folder / "cellector" / f"{name}_criteria.npy", save_criteria)


# fullRedIdx = np.concatenate(self.redIdx)
# fullManualLabels = np.stack((np.concatenate(self.manualLabel), np.concatenate(self.manualLabelActive)))
# self.redCell.saveone(fullRedIdx, "mpciROIs.redCellIdx")
# self.redCell.saveone(fullManualLabels, "mpciROIs.redCellManualAssignments")
# for idx, name in enumerate(self.roi_processor.features):
#     cFeatureCutoffs = self.featureCutoffs[idx]
#     if not (self.feature_active[idx][0]):
#         cFeatureCutoffs[0] = np.nan
#     if not (self.feature_active[idx][1]):
#         cFeatureCutoffs[1] = np.nan
#     self.redCell.saveone(self.featureCutoffs[idx], self.redCell.oneNameFeatureCutoffs(name))

# print(f"Red Cell curation choices are saved for session {self.redCell.sessionPrint()}")
