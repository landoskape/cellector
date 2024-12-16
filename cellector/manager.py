from typing import Union, List, Optional
from pathlib import Path
from copy import copy
import numpy as np
from . import io


class CellectorManager:
    def __init__(self, root_dir: Union[Path, str], include_features: Optional[List[str]] = None, exclude_features: Optional[List[str]] = None):
        self.root_dir = Path(root_dir)
        self.include_features = include_features
        self.exclude_features = exclude_features or []

        # Identify feature files and criteria files
        self.feature_names = io.identify_feature_files(self.root_dir, criteria=False)
        self.criteria_names = io.identify_feature_files(self.root_dir, criteria=True)

        # Make the inclusion list the same as the feature names if not provided
        if self.include_features is None:
            self.include_features = self.feature_names

        # Load feature data
        self.features = {feature: io.load_saved_feature(self.root_dir, feature) for feature in self.feature_names}
        self.criteria = {criteria: io.load_saved_criteria(self.root_dir, criteria) for criteria in self.criteria_names}

        # Populate missing criteria with None
        for feature in self.feature_names:
            if feature not in self.criteria:
                # Initial criteria is None, None, meaning don't use a min or max cutoff
                self.criteria[feature] = np.array([None, None])

        # Set the number of ROIs
        self.num_rois = self.features[self.feature_names[0]].shape[0]

        # Data checks and numROIs
        if not all([f.shape[0] == self.num_rois for f in self.features.values()]):
            raise ValueError("Feature arrays have different numbers of ROIs!")
        if not all([f.ndim == 1 for f in self.features.values()]):
            raise ValueError("Feature arrays have incorrect shape!")
        if not all([len(c) == 2 and c.ndim == 1 for c in self.criteria.values()]):
            raise ValueError("Feature criteria arrays have incorrect shape!")

        # Load manual selection data (or initialize to empty if not saved)
        if io.is_manual_selection_saved(self.root_dir):
            self.manual_label, self.manual_label_active = io.load_manual_selection(self.root_dir)
        else:
            # Initial state is for manual labels to all be False, but for none of them to be active
            self.manual_label = np.zeros(self.num_rois, dtype=bool)
            self.manual_label_active = np.zeros(self.num_rois, dtype=bool)

        # Initialize idx_meets_criteria
        self.compute_idx_meets_criteria()

    def save_criteria(self):
        """Save the criteria values to disk."""
        for feature, criteria in self.criteria.items():
            io.save_criteria(self.root_dir, feature, criteria)

    def save_manual_selection(self):
        """Save the manual selection labels to disk."""
        io.save_manual_selection(self.root_dir, self.manual_label, self.manual_label_active)

    def save_selected(self):
        """Save the indices of selected ROIs to disk."""
        self.compute_idx_meets_criteria()
        idx_selected = copy(self.idx_meets_criteria)
        idx_selected[self.manual_label_active] = self.manual_label[self.manual_label_active]
        io.save_selection(self.root_dir, idx_selected)

    def save_all(self):
        """Save all data to disk."""
        self.save_criteria()
        self.save_manual_selection()
        self.save_selected()

    def compute_idx_meets_criteria(self):
        """Compute the index of cells meeting the criteria defined by the features."""
        self.idx_meets_criteria = np.full(self.num_rois, True)
        for feature, value in self.features.items():
            if feature in self.include_features and feature not in self.exclude_features:
                criteria = self.criteria[feature]
                if criteria[0] is not None:
                    self.idx_meets_criteria &= value >= criteria[0]
                if criteria[1] is not None:
                    self.idx_meets_criteria &= value <= criteria[1]

    def update_criteria(self, feature: str, criteria: np.ndarray):
        """Update the criteria for a particular feature.

        Parameters
        ----------
        feature : str
            Feature name.
        criteria : np.ndarray
            Array of shape (2,) with the minimum and maximum criteria or None for no criteria.
        """
        if feature not in self.feature_names:
            raise ValueError(f"Feature {feature} not found in the feature list!")

        if len(criteria) != 2 or not isinstance(criteria, np.ndarray) or criteria.ndim != 1:
            raise ValueError(f"Criteria should be a numpy array with shape (2,)")

        self.criteria[feature] = criteria
        self.compute_idx_meets_criteria()  # Update this to reflect the new change

    def update_manual_labels(self, idx_roi: int, label: bool, active: bool):
        """Update the manual labels for an ROI.

        Parameters
        ----------
        idx_roi : int
            Index of the ROI to update.
        label: bool
            Whether the ROI is selected or not.
        active : np.ndarray
            Whether to consider the manual label for this ROI.
        """
        self.manual_label[idx_roi] = label
        self.manual_label_active[idx_roi] = active
