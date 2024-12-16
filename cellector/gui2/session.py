# This is going to go somewhere else, but since I'm refactoring the GUI class and moving
# common functionality here, I'll keep it in the gui2 folder for now. 

from typing import Union, Dict, Optional
from pathlib import Path
import numpy as np
from ..roi_processor import RoiProcessor
from .. import io


def update_by_feature_criterion(self):
    """Update the idx of cells meeting the criterion defined by the features."""
    # start with all as targets
    self.idx_meets_criteria = np.full(self.roi_processor.num_rois, True)
    for feature, value in self.roi_processor.features.items():
        if self.feature_active[feature][0]:
            # only keep in idx_meets_criteria if above minimum
            self.idx_meets_criteria &= value >= self.feature_cutoffs[feature][0]
        if self.feature_active[feature][1]:
            # only keep in idx_meets_criteria if below maximum
            self.idx_meets_criteria &= value <= self.feature_cutoffs[feature][1]

    self.regenerate_mask_data()