from typing import Union, List, Optional
from pathlib import Path
import shutil
import re
import numpy as np
from .base import get_save_directory
from .base import criteria_path, save_criteria, FEATURE_EXTENSION, CRITERIA_EXTENSION
from ..utils import deprecated


def clear_cellector_files(root_dir: Union[Path, str]) -> None:
    """Clear all files in the cellector save directory for a root directory.

    Parameters
    ----------
    root_dir : Path or str
        Path to the root directory.
    """
    save_dir = get_save_directory(root_dir)
    if save_dir.exists():
        for file in save_dir.glob("*"):
            file.unlink()
        save_dir.rmdir()


def identify_feature_files(
    root_dir: Union[Path, str], criteria: bool = True
) -> List[str]:
    """Identify cellector features or feature criteria files that are saved in a cellector directory.

    Features and feature criteria are saved with a particular convention that makes them
    easy to find, (features: {name}_feature.npy, criteria: {name}_featurecriteria.npy).
    This function will return a list of the names of the features or feature criteria
    that are saved in the cellector directory under root_dir.

    Parameters
    ----------
    root_dir : Path or str
        Path to the root directory (cellector folder is root_dir / 'cellector').
    criteria : bool, optional
        Whether to identify feature criteria files (instead of features), by default True

    Returns
    -------
    feature_names : list of str
        List of feature (or feature criteria) names that are saved in the cellector folder.
    """
    EXTENSION = CRITERIA_EXTENSION if criteria else FEATURE_EXTENSION
    save_dir = get_save_directory(root_dir)
    files = [pth.stem for pth in save_dir.glob(f"*{EXTENSION}.npy")]
    names = [re.match(f"(.*){EXTENSION}", f).group(1) for f in files]
    return names


def propagate_criteria(root_dir: Union[Path, str], *target_dirs: Union[Path, str]):
    """Copy feature criteria saved under root_dir to other directories.

    Parameters
    ----------
    root_dir : Path or str
        Path to the root directory where the feature criteria are saved.
    target_dirs : list of Path or str
        List of directories to copy the feature criteria to.

    Returns
    -------
    successful_copies : dict
        Dictionary of successful copies with the target directory as the key and a list of copied files as the value.
    unsuccessful_copies : dict
        Dictionary of unsuccessful copies with the target directory as the key and the error as the value.
    """
    if not target_dirs:
        raise ValueError("No directories to copy feature criteria to!")

    # Load feature criteria names from root_dir
    criteria_names = identify_feature_files(root_dir, criteria=True)
    # Load feature criteria values from root_dir
    criteria_values = {
        name: np.load(criteria_path(root_dir, name)) for name in criteria_names
    }

    # Copy feature criteria to target_dirs
    successful_copies = {}
    unsuccessful_copies = {}
    for target_dir in target_dirs:
        successful_copies[target_dir] = []
        try:
            for name, values in zip(criteria_names, criteria_values):
                save_criteria(target_dir, name, values)
                successful_copies[target_dir].append(name)
        except Exception as e:
            # remove incomplete files from failed copy
            for name in successful_copies[target_dir]:
                criteria_path(target_dir, name).unlink()
            unsuccessful_copies[target_dir] = e
            successful_copies.pop(target_dir)
    return successful_copies, unsuccessful_copies


def identify_cellector_folders(top_level_dir: Union[Path, str]):
    """Identify any directories that contain cellector save directories.

    Will return a list of paths that contain the "cellector" directory one below. The
    search is recursive but constrained by top_level_dir so you don't search your entire
    path if you don't need to.

    Parameters
    ----------
    top_level_dir : Path or str
        Path to the top level directory to search for cellector save directories.
    """
    import os

    top_level_dir = Path(top_level_dir)
    cellector_dirs = []
    # recursively search for directories with a "cellector" directory
    for root, dirs, _ in os.walk(top_level_dir):
        if "cellector" in dirs:
            cellector_dirs.append(Path(root))
    return cellector_dirs


# @deprecated("Provided to address backwards incompatibility")
def update_feature_paths(root_dirs: List[Union[Path, str]], remove_old: bool = True):
    """Update the feature paths for a feature across multiple root directories.

    Upon changing from version 0.1.0 to 0.2.0, the feature paths were updated to include
    the "_feature" suffix. This function updates the feature paths for a feature across
    multiple root directories. It depends on

    Parameters
    ----------
    root_dirs : list of Path or str
        List of root directories to update the feature paths for.
    remove_old : bool, optional
        Whether to remove the old feature files after updating, by default True.
    """
    from .base import criteria_path, feature_path

    move_method = shutil.move if remove_old else shutil.copy2

    def _identify_saved_criteria(root_dir):
        """Identify any feature criteria that have been saved to disk."""
        save_dir = get_save_directory(root_dir)
        features = [pth.stem for pth in save_dir.glob("*_criteria.npy")]
        feature_matches = [re.match("(.*)_criteria", f) for f in features]
        criteria_names = [m.group(1) for m in feature_matches if m]
        return criteria_names

    def _identify_saved_features(root_dir):
        """Identify features that have been saved to disk. This will look for features
        that are associated with a criteria file and will miss any features that have
        been saved but don't have a criteria file due to the poor decision of the first
        naming convention.
        """
        save_dir = get_save_directory(root_dir)
        criteria_names = _identify_saved_criteria(root_dir)
        feature_names = [
            cname for cname in criteria_names if (save_dir / f"{cname}.npy").exists()
        ]
        return feature_names

    def _old_criteria_path(save_dir, name):
        return save_dir / f"{name}_criteria.npy"

    def _old_feature_path(save_dir, name):
        return save_dir / f"{name}.npy"

    for root_dir in root_dirs:
        criteria_names = _identify_saved_criteria(root_dir)
        feature_names = _identify_saved_features(root_dir)
        save_dir = get_save_directory(root_dir)
        for criteria_name in criteria_names:
            old_path = _old_criteria_path(save_dir, criteria_name)
            new_path = criteria_path(save_dir, criteria_name)
            move_method(old_path, new_path)
        for feature_name in feature_names:
            old_path = _old_feature_path(save_dir, feature_name)
            new_path = feature_path(save_dir, feature_name)
            move_method(old_path, new_path)


# @deprecated("Provided to address backwards incompatibility with manual selection shape")
def update_manual_selection_shape(root_dirs: List[Union[Path, str]]):
    """Update the feature paths for a feature across multiple root directories.

    After version 0.2.0, the handling of manual selections data was changed to stack them
    on axis=0 for simplification. This function updates the manual selection data across
    multiple root directories to be consistent with the new format. If any
    manual_selection data is found with shape (num_rois, 2), it will transpose it and
    resave on the same path.

    Parameters
    ----------
    root_dirs : list of Path or str
        List of root directories to update the manual selection shape for.
    """
    from .base import is_manual_selection_saved, manual_selection_path

    for root_dir in root_dirs:
        if is_manual_selection_saved(root_dir):
            filepath = manual_selection_path(root_dir)
            manual_selection = np.load(filepath)
            if manual_selection.ndim != 2:
                print(
                    f"Manual selection data in {root_dir} has shape {manual_selection.shape}, expected (2, num_rois) or (num_rois, 2). Skipping."
                )
                continue
            if manual_selection.shape[0] != 2 and manual_selection.shape[1] == 2:
                # Wrong shape, need to transpose
                manual_selection = manual_selection.T
                np.save(filepath, manual_selection)
            if manual_selection.shape[0] == 2:
                # Correct shape, do nothing
                pass
            else:
                print(
                    f"Manual selection data in {root_dir} has shape {manual_selection.shape}, expected (2, num_rois) or (num_rois, 2). Skipping."
                )


# @deprecated("Provided to address an update with the standard name for the selected cells.")
def update_idx_selection_filenames(root_dirs: List[Union[Path, str]]):
    """Update the idx_selection filenames from targetcell.npy to idx_selected.npy.

    After version 1.0.0, the idx_selection filenames were updated to idx_selected.npy to
    be more clear with the purpose of the file and less confusing when using the GUI.
    This function updates the filenames across multiple root directories from the
    previous name targetcell.npy to the new name idx_selected.npy.

    Parameters
    ----------
    root_dirs : list of Path or str
        List of root directories to update the targetcell filenames for.
    """
    from .base import idx_selected_path, get_save_directory

    for root_dir in root_dirs:
        targetcell_path = get_save_directory(root_dir) / "targetcell.npy"
        if targetcell_path.exists():
            new_path = idx_selected_path(root_dir)
            shutil.move(targetcell_path, new_path)
