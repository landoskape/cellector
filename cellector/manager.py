from typing import Union, List, Optional
from pathlib import Path
from copy import copy
import numpy as np
from . import io
from .roi_processor import RoiProcessor


class CellectorManager:
    def __init__(
        self,
        root_dir: Union[Path, str],
        exclude_features: Optional[List[str]] = None,
        num_rois: Optional[int] = None,
    ):
        """Initialize CellectorManager.

        Parameters
        ----------
        root_dir : Union[Path, str]
            Path to root directory for saving/loading data
        exclude_features : Optional[List[str]]
            List of feature names to exclude from selection criteria
        num_rois : Optional[int]
            Number of ROIs to expect in the data. If not provided, the number of ROIs
            will be determined by the first feature file found in the root directory.
        """
        self.root_dir = Path(root_dir)
        self.exclude_features = exclude_features or []
        if num_rois is not None:
            self.num_rois = num_rois

        # Identify feature files and criteria files
        feature_names = io.identify_feature_files(self.root_dir, criteria=False)

        # Load feature data and associated criteria
        self.features = {}
        self.criteria = {}
        for feature in feature_names:
            self.add_feature(feature, io.load_feature(self.root_dir, feature))

        # Load or initialize manual selection data
        if io.is_manual_selection_saved(self.root_dir):
            self.manual_label, self.manual_label_active = io.load_manual_selection(
                self.root_dir
            )
        else:
            # Initial state is for manual labels to all be set to False, but for none of them to be active
            self.manual_label = np.zeros(self.num_rois, dtype=bool)
            self.manual_label_active = np.zeros(self.num_rois, dtype=bool)

    @classmethod
    def make_from_roi_processor(
        cls,
        roi_processor: RoiProcessor,
        exclude_features: Optional[List[str]] = None,
    ) -> "CellectorManager":
        """Create a CellectorManager from an existing RoiProcessor.

        CellectorManager is usually initialized from disk, but this method allows the
        user to create a new manager from an existing RoiProcessor object, which will
        include any features / criteria that are stored on disk and add any features that
        are in the RoiProcessor but not on disk yet. This is useful when pipelining the
        and using the CellectorManager class to store the results of processing a session

        Parameters
        ----------
        roi_processor : RoiProcessor
            RoiProcessor object to include features from and define the root directory.
        exclude_features : Optional[List[str]]
            List of feature names to exclude from selection criteria

        Returns
        -------
        CellectorManager
            Manager object with features and criteria from the RoiProcessor object.
        """
        manager = cls(roi_processor.root_dir, exclude_features, roi_processor.num_rois)
        for feature_name, feature_values in roi_processor.features.items():
            if feature_name not in manager.features:
                manager.add_feature(feature_name, feature_values)
        return manager

    def add_feature(self, feature_name, feature_values):
        """Add a new feature to the manager.

        Parameters
        ----------
        feature_name : str
            Name of the feature to add.
        feature_values : np.ndarray
            Array of shape (num_rois,) with the feature values.
        """
        if not hasattr(self, "num_rois"):
            self.num_rois = feature_values.shape[0]

        if feature_values.shape[0] != self.num_rois or feature_values.ndim != 1:
            raise ValueError(
                f"Feature array has incorrect number of ROIs! Expected {self.num_rois}, received {feature_values.shape[0]}."
            )

        self.features[feature_name] = feature_values

        # Initialize criteria for this feature
        if io.is_criteria_saved(self.root_dir, feature_name):
            # Load criteria if it exists
            self.criteria[feature_name] = io.load_criteria(self.root_dir, feature_name)
        else:
            # Initial criteria is None, None, meaning don't use a min or max cutoff
            self.criteria[feature_name] = np.array([None, None])

        # Update the index of cells meeting the criteria whenever adding a new feature
        self.compute_idx_meets_criteria()

    def compute_idx_meets_criteria(self):
        """Compute the index of cells meeting the criteria defined by the features."""
        self.idx_meets_criteria = np.full(self.num_rois, True)
        for feature, value in self.features.items():
            if feature not in self.exclude_features:
                criteria = self.criteria[feature]
                if criteria[0] is not None:
                    self.idx_meets_criteria &= value >= criteria[0]
                if criteria[1] is not None:
                    self.idx_meets_criteria &= value <= criteria[1]

    def compute_idx_selected(self):
        """Compute the index of cells that are selected."""
        self.compute_idx_meets_criteria()
        idx_selected = copy(self.idx_meets_criteria)
        idx_selected[self.manual_label_active] = self.manual_label[
            self.manual_label_active
        ]
        return idx_selected

    def update_criteria(self, feature: str, criterion: Union[List, np.ndarray]):
        """Update the criteria for a particular feature.

        Parameters
        ----------
        feature : str
            Feature name.
        criterion : np.ndarray
            Array of shape (2,) with the minimum and maximum criteria or None for no criteria.
        """
        if feature not in self.features or feature not in self.criteria:
            raise ValueError(
                f"Feature {feature} not found in the feature/criteria lists!"
            )

        if len(criterion) != 2:
            raise ValueError(
                f"Criterion should be a list or numpy array with shape (2,)"
            )

        if (
            criterion[0] is not None
            and criterion[1] is not None
            and criterion[0] > criterion[1]
        ):
            raise ValueError(
                f"Minimum criterion should be less than maximum criterion!"
            )

        self.criteria[feature] = np.array(criterion)
        self.compute_idx_meets_criteria()  # Update this to reflect the new change

    def update_manual_labels(self, idx_roi: int, label: bool, active: bool):
        """Update the manual labels for an ROI.

        Parameters
        ----------
        idx_roi : int
            Index of the ROI to update.
        label: bool
            Whether the ROI is selected or not.
        active : bool
            Whether to consider the manual label for this ROI.
        """
        self.manual_label[idx_roi] = label
        self.manual_label_active[idx_roi] = active

    def save_features(self):
        """Save the feature values to disk."""
        for feature, values in self.features.items():
            io.save_feature(self.root_dir, feature, values)

    def save_criteria(self):
        """Save the criteria values to disk."""
        for feature, criteria in self.criteria.items():
            io.save_criteria(self.root_dir, feature, criteria)

    def save_manual_selection(self):
        """Save the manual selection labels to disk."""
        io.save_manual_selection(
            self.root_dir, self.manual_label, self.manual_label_active
        )

    def save_idx_selected(self):
        """Save the indices of selected ROIs to disk."""
        io.save_idx_selected(self.root_dir, self.compute_idx_selected())

    def save_all(self):
        """Save all data to disk."""
        self.save_features()
        self.save_criteria()
        self.save_manual_selection()
        self.save_idx_selected()
