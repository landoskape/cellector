from typing import Union, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from ..roi_processor import RoiProcessor

FEATURE_EXTENSION = "_feature"
CRITERIA_EXTENSION = "_featurecriteria"


def get_save_directory(root_dir: Union[Path, str]) -> Path:
    """Get the cellector save directory from a root folder.

    Parameters
    ----------
    root_dir : Path or str
        Path to the root directory.

    Returns
    -------
    save_dir : Path
        Path to the save directory for the root directory.
    """
    return Path(root_dir) / "cellector"


def feature_path(root_dir: Union[Path, str], name: str) -> Path:
    """Get the path to a feature file.

    Parameters
    ----------
    root_dir : Path or str
        Path to the root directory.
    name : str
        Name of the feature.

    Returns
    -------
    path : Path
        Path to the feature file.
    """
    save_dir = get_save_directory(root_dir)
    return save_dir / f"{name}{FEATURE_EXTENSION}.npy"


def save_feature(root_dir: Union[Path, str], name: str, feature: np.ndarray) -> None:
    """Save a feature to disk.

    Parameters
    ----------
    root_dir : Path or str
        Path to the root directory.
    name : str
        Name of the feature to save.
    feature : np.ndarray
        Feature data to save to disk.
    """
    save_dir = get_save_directory(root_dir)
    save_dir.mkdir(exist_ok=True)
    np.save(feature_path(root_dir, name), feature)


def load_saved_feature(root_dir: Union[Path, str], name: str) -> np.ndarray:
    """Load a feature from disk.

    Parameters
    ----------
    root_dir : Path or str
        Path to the root directory.
    name : str
        Name of the feature to load.

    Returns
    -------
    feature : np.ndarray
        Feature data loaded from disk.
    """
    return np.load(feature_path(root_dir, name))


def is_feature_saved(root_dir: Union[Path, str], name: str) -> bool:
    """Check if a feature exists on disk.

    Parameters
    ----------
    root_dir : Path or str
        Path to the root directory.
    name : str
        Name of the feature to check.

    Returns
    -------
    exists : bool
        Whether the feature exists on disk.
    """
    return feature_path(root_dir, name).exists()


def criteria_path(root_dir: Union[Path, str], name: str) -> Path:
    """Get the path to a feature criterion file.

    Parameters
    ----------
    root_dir : Path or str
        Path to the root directory.
    name : str
        Name of the feature criterion.

    Returns
    -------
    path : Path
        Path to the feature criterion file.
    """
    save_dir = get_save_directory(root_dir)
    return save_dir / f"{name}{CRITERIA_EXTENSION}.npy"


def save_criteria(root_dir: Union[Path, str], name: str, criteria: np.ndarray) -> None:
    """Save a feature criterion to disk.

    Parameters
    ----------
    root_dir : Path or str
        Path to the root directory.
    name : str
        Name of the feature criterion to save.
    criteria : np.ndarray
        Criterion data to save to disk.
    """
    save_dir = get_save_directory(root_dir)
    save_dir.mkdir(exist_ok=True)
    np.save(criteria_path(root_dir, name), criteria)


def load_saved_criteria(root_dir: Union[Path, str], name: str) -> np.ndarray:
    """Load a feature criterion from disk.

    Parameters
    ----------
    root_dir : Path or str
        Path to the root directory.
    name : str
        Name of the feature criterion to load.

    Returns
    -------
    criterion : np.ndarray
        Criterion data loaded from disk.
    """
    return np.load(criteria_path(root_dir, name), allow_pickle=True)


def is_criteria_saved(root_dir: Union[Path, str], name: str) -> bool:
    """Check if a feature criterion exists on disk.

    Parameters
    ----------
    root_dir : Path or str
        Path to the root directory.
    name : str
        Name of the feature criterion to check.

    Returns
    -------
    exists : bool
        Whether the feature criterion exists on disk.
    """
    return criteria_path(root_dir, name).exists()


def manual_selection_path(root_dir: Union[Path, str]) -> Path:
    """Get the path to the manual selection labels file.

    Parameters
    ----------
    root_dir : Path or str
        Path to the root directory.

    Returns
    -------
    path : Path
        Path to the manual selection labels file.
    """
    save_dir = get_save_directory(root_dir)
    return save_dir / "manual_selection.npy"


def save_manual_selection(root_dir: Union[Path, str], manual_labels: np.ndarray, manual_labels_active: np.ndarray) -> None:
    """Save manual selection labels to disk.

    Parameters
    ----------
    root_dir : Path or str
        Path to the root directory.
    manual_labels : np.ndarray
        Manual labels to save to disk.
    manual_labels_active : np.ndarray
        Manual label active flags to save to disk.
    """
    save_dir = get_save_directory(root_dir)
    save_dir.mkdir(exist_ok=True)
    manual_selection = np.stack((manual_labels, manual_labels_active))
    np.save(manual_selection_path(root_dir), manual_selection)


def load_manual_selection(root_dir: Union[Path, str]) -> Tuple[np.ndarray, np.ndarray]:
    """Load manual selection labels from disk.

    Parameters
    ----------
    root_dir : Path or str
        Path to the root directory.

    Returns
    -------
    manual_label : np.ndarray
        Manual label array.
    manual_label_active : np.ndarray
        Manual label active flag array.
    """
    manual_selection = np.load(manual_selection_path(root_dir))
    manual_label = manual_selection[0]
    manual_label_active = manual_selection[1]
    return manual_label, manual_label_active


def is_manual_selection_saved(root_dir: Union[Path, str]) -> bool:
    """Check if manual selection labels exist on disk.

    Parameters
    ----------
    root_dir : Path or str
        Path to the root directory.

    Returns
    -------
    exists : bool
        Whether manual selection labels exist on disk.
    """
    return manual_selection_path(root_dir).exists()


def idx_selected_path(root_dir: Union[Path, str]) -> Path:
    """Get the path to the selected ROI indices file.

    Parameters
    ----------
    root_dir : Path or str
        Path to the root directory.

    Returns
    -------
    path : Path
        Path to the selected ROI indices file.
    """
    save_dir = get_save_directory(root_dir)
    return save_dir / "idx_selected.npy"


def save_idx_selected(root_dir: Union[Path, str], idx_selected: np.ndarray) -> None:
    """Save selected ROI indices to disk.

    Parameters
    ----------
    root_dir : Path or str
        Path to the root directory.
    idx_selected : np.ndarray
        Selected ROI indices to save to disk.
    """
    save_dir = get_save_directory(root_dir)
    save_dir.mkdir(exist_ok=True)
    np.save(idx_selected_path(root_dir), idx_selected)


def load_idx_selected(root_dir: Union[Path, str]) -> np.ndarray:
    """Load selected ROI indices from disk.

    Parameters
    ----------
    root_dir : Path or str
        Path to the root directory.

    Returns
    -------
    idx_selected : np.ndarray
        Selected ROI indices loaded from disk.
    """
    return np.load(idx_selected_path(root_dir))


def is_idx_selected_saved(root_dir: Union[Path, str]) -> bool:
    """Check if selected ROI indices exist on disk.

    Parameters
    ----------
    root_dir : Path or str
        Path to the root directory.

    Returns
    -------
    exists : bool
        Whether selected ROI indices exist on disk.
    """
    return idx_selected_path(root_dir).exists()


def save_selection(
    roi_processor: RoiProcessor,
    idx_target: np.ndarray,
    criteria: Dict[str, list],
    manual_selection: Optional[np.ndarray] = None,
):
    """Save roi processor features, criterion, and target index to disk.

    Parameters
    ----------
    roi_processor : RoiProcessor
        RoiProcessor object with features and folders to save to.
    idx_target : np.ndarray
        Target index over all ROIs. Should be a numpy array with shape (num_rois,) where each value is a boolean indicating
        whether the ROI is one of the "target" cells. This means it meets all feature criteria and manual selection criteria.
    criteria : Dict[str, list]
        Dictionary of feature criteria for each feature. Each value in the dictionary should be a 2 element list containing
        the minimum and maximum values for the feature. If the minimum or maximum cutoff is ignored, then that value should
        be set to None.
    manual_selection : np.ndarray
        Manual selection labels for each ROI. Shape should be (num_rois, 2), where the first column is the manual label
        and the second column is whether or not to use a manual label for that cell.
    """
    # Check that everything has the expected shapes
    if idx_target.shape[0] != roi_processor.num_rois:
        raise ValueError(f"Target indices have shape {idx_target.shape} but expected {roi_processor.num_rois}!")
    if manual_selection is not None:
        if (manual_selection.shape[0] != roi_processor.num_rois) or (manual_selection.shape[1] != 2):
            raise ValueError(f"Manual selection labels have shape {manual_selection.shape} but expected ({roi_processor.num_rois}, 2)!")
    for name, value in criteria.items():
        if name not in roi_processor.features:
            raise ValueError(f"Feature {name} not found in roi_processor features!")
        if len(value) != 2:
            raise ValueError(f"Feature criteria {name} has shape {value.shape} but expected (2,)!")
    if any(feature not in criteria for feature in roi_processor.features):
        raise ValueError(f"Feature criteria missing for features: {set(roi_processor.features) - set(criteria)}!")

    # Load and create save directory
    save_dir = get_save_directory(roi_processor.root_dir)
    save_dir.mkdir(exist_ok=True)

    # Save features values for each plane
    for name, values in roi_processor.features.items():
        save_feature(roi_processor.root_dir, name, values)
    if manual_selection is not None:
        save_manual_selection(roi_processor.root_dir, manual_selection)

    # Save selection indices
    np.save(save_dir / "targetcells.npy", idx_target)

    # Save feature criteria
    for name, value in criteria.items():
        save_criteria(roi_processor.root_dir, name, value)
