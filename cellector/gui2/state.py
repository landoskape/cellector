class SelectionState:
    def __init__(self):
        self.show_control_cells = False  # show control cells instead of target cells
        self.show_mask_image = False  # if true, will show mask image, if false, will show mask labels
        self.mask_visibility = True  # if true, will show either mask image or label, otherwise will not show either!
        self.use_manual_labels = True  # if true, then will apply manual labels after using features to compute idx_meets_criteria
        self.only_manual_labels = False  # if true, only show manual labels of selected category...
        self.color_state = 0  # indicates which color to display maskLabels (0:random, 1-4:color by feature)
        self.color_state_names = ["random", *self.roi_processor.features.keys()]
        self.idx_colormap = 0  # which colormap to use for pseudo coloring the masks
        self.colormaps = ["plasma", "autumn", "spring", "summer", "winter", "hot"]


class SelectionData:
    def __init__(self, roi_processor):
        self.roi_processor = roi_processor
