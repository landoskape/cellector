from typing import Union, List
from pathlib import Path
import shutil
import re
from .base import get_save_directory
from ..utils import deprecated


def clear_cellector_files(root_dir: Union[Path, str]):
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

    save_dir = get_save_directory(root_dir)
    copy_dirs = [get_save_directory(target_dir) for target_dir in target_dirs]
    successful_copies = {}
    unsuccessful_copies = {}
    for copy_dir in copy_dirs:
        copy_dir.mkdir(exist_ok=True)
        successful_copies[copy_dir] = []
        try:
            for file in save_dir.glob("*_criteria.npy"):
                shutil.copy(file, copy_dir / file.name)
                successful_copies[copy_dir].append(file.name)
        except Exception as e:
            # remove incomplete files from failed copy
            for file in successful_copies[copy_dir]:
                (copy_dir / file).unlink()
            unsuccessful_copies[copy_dir] = e
    return successful_copies, unsuccessful_copies


@deprecated("Provided to address backwards incompatibility", version="1.0.0")
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


@deprecated("Provided to address backwards incompatibility", version="1.0.0")
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
        feature_names = [cname for cname in criteria_names if (save_dir / f"{cname}.npy").exists()]
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
