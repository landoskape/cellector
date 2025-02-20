from copy import copy
import functools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# GUI-related modules
import napari
from napari.utils.colormaps import label_colormap, direct_colormap
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QGraphicsProxyWidget, QPushButton

from ..roi_processor import RoiProcessor
from ..utils import split_planes, deprecated
from .. import io

basic_button_style = """
QWidget {
    background-color: #1F1F1F;
    color: #F0F0F0;
    font-family: Arial, sans-serif;
}

QPushButton:hover {
    background-color: #45a049;
    font-size: 10px;
    font-weight: bold;
    border: none;
    border-radius: 5px;
    padding: 5px 5px;
}
"""

q_checked_style = """
QWidget {
    background-color: #1F1F1F;
    color: red;
    font-family: Arial, sans-serif;
}
"""

q_not_checked_style = """
QWidget {
    background-color: #1F1F1F;
    color: #F0F0F0;
    font-family: Arial, sans-serif;
}
"""


@deprecated(
    "This is the old version of the SelectionGUI class. Use the new version with from cellector.gui import SelectionGUI."
)
class SelectionGUI:
    """A GUI for selecting cells based on features in reference to a fluorescence image.

    This GUI allows the user to interactively select cells that meet criterion based on
    filters computed for each cell. The GUI shows histograms of the intensity features of
    the cells and allows the user to select the range of feature values that qualify as
    target cells. The GUI also allows the user to manually label cells as target or control.

    Parameters
    ----------
    roi_processor : RoiProcessor
        An instance of the RoiProcessor class that contains the masks and fluorescence data.
    num_bins : int, optional
        The number of bins to use for the histograms of the intensity features. Default is 50.
    """

    def __init__(self, roi_processor: RoiProcessor, num_bins=50):
        if not isinstance(roi_processor, RoiProcessor):
            raise ValueError(
                "roi_processor must be an instance of the RoiProcessor class."
            )

        self.roi_processor = roi_processor
        self.num_bins = num_bins
        self.plane_idx = (
            0  # determines which plane is currently being shown in the napari viewer
        )

        self.num_features = len(self.roi_processor.features)
        self.feature_active = {key: [True, True] for key in self.roi_processor.features}

        # process initial plane
        self.idx_meets_criteria = np.full(self.roi_processor.num_rois, True)

        if io.is_manual_selection_saved(self.roi_processor.root_dir):
            self.manual_label, self.manual_label_active = io.load_manual_selection(
                self.roi_processor.root_dir
            )
        else:
            self.manual_label = np.full(self.roi_processor.num_rois, False)
            self.manual_label_active = np.full(self.roi_processor.num_rois, False)

        # open napari viewer and associated GUI features
        self.show_control_cells = False  # show control cells instead of target cells
        self.show_mask_image = (
            False  # if true, will show mask image, if false, will show mask labels
        )
        self.mask_visibility = True  # if true, will show either mask image or label, otherwise will not show either!
        self.use_manual_labels = True  # if true, then will apply manual labels after using features to compute idx_meets_criteria
        self.only_manual_labels = (
            False  # if true, only show manual labels of selected category...
        )
        self.color_state = 0  # indicates which color to display maskLabels (0:random, 1-4:color by feature)
        self.color_state_names = ["random", *self.roi_processor.features.keys()]
        self.idx_colormap = 0  # which colormap to use for pseudo coloring the masks
        self.colormaps = ["plasma", "autumn", "spring", "summer", "winter", "hot"]
        self._initialize_napari_viewer()

    # ------------------------------------------------------------------------------------------------
    # ---------------------------------- Central Handling Elements -----------------------------------
    # ------------------------------------------------------------------------------------------------
    def update_text(self, text):
        """Central method for updating text in the text area and on the status area."""
        self.text_area.setText(text)
        self.viewer.status = text

    def update_visibility(self):
        """Update the visibility of the masks and labels in the napari viewer."""
        self.masks.visible = self.show_mask_image and self.mask_visibility
        self.labels.visible = not self.show_mask_image and self.mask_visibility

    def update_feature_plots(self):
        """Update the histograms of the intensity features of the target cells in the napari viewer."""
        for feature in self.roi_processor.features:
            self.hist_graphs[feature].setOpts(
                height=self.h_values_full[feature][self.plane_idx]
            )
            self.hist_selected[feature].setOpts(
                height=self.h_values_selected[feature][self.plane_idx]
            )

    def update_label_colors(self):
        """Update the colors of the labels in the napari viewer."""
        color_state_name = self.color_state_names[self.color_state]
        if color_state_name == "random":
            # this is inherited from the default random colormap in napari
            colormap = label_colormap(49, 0.5, background_value=0)
        else:
            # assign colors based on the feature values for every ROI
            norm = mpl.colors.Normalize(
                vmin=self.feature_range[color_state_name][0],
                vmax=self.feature_range[color_state_name][1],
            )
            colors = plt.colormaps[self.colormaps[self.idx_colormap]](
                norm(self.roi_processor.features[color_state_name])
            )
            color_dict = dict(zip(1 + np.arange(self.roi_processor.num_rois), colors))
            color_dict[None] = np.array(
                [0.0, 0.0, 0.0, 0.0], dtype=np.single
            )  # transparent background (or default)
            colormap = direct_colormap(color_dict)
        # Update colors of the labels
        self.labels.colormap = colormap

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

    def regenerate_mask_data(self):
        """Regenerate the mask image and labels in the napari viewer based on the current selection.

        Used whenever the selection of cells is updated or the manual labels are updated such that
        the GUI reflects the current selection appropriately.
        """
        self.masks.data = self.mask_image
        self.labels.data = self.mask_labels
        features_by_plane = {
            key: split_planes(value, self.roi_processor.rois_per_plane)
            for key, value in self.roi_processor.features.items()
        }
        idx_selected_by_plane = split_planes(
            self.idx_selected, self.roi_processor.rois_per_plane
        )
        for feature in self.roi_processor.features:
            for iplane in range(self.roi_processor.num_planes):
                c_feature_values = features_by_plane[feature][iplane][
                    idx_selected_by_plane[iplane]
                ]
                self.h_values_selected[feature][iplane] = np.histogram(
                    c_feature_values, bins=self.h_bin_edges[feature]
                )[0]

        # regenerate histograms
        for feature in self.roi_processor.features:
            self.hist_selected[feature].setOpts(
                height=self.h_values_selected[feature][self.plane_idx]
            )

    def save_selection(self):
        """Save the current selection of cells to files."""
        manual_selection = np.stack((self.manual_label, self.manual_label_active)).T
        feature_criteria = {}
        for feature in self.roi_processor.features:
            feature_criteria[feature] = self.feature_cutoffs[feature]
            if not self.feature_active[feature][0]:
                feature_criteria[feature][0] = None
            if not self.feature_active[feature][1]:
                feature_criteria[feature][1] = None
        io.save_selection(
            self.roi_processor,
            self.idx_target,
            feature_criteria,
            manual_selection=manual_selection,
        )
        self.update_text("Selection saved!")

    # ------------------------------
    # --------- properties ---------
    # ------------------------------
    @property
    def mask_image(self):
        """Return the masks in each volume as an image by summing the selected masks in each plane.

        Each pixel in the image is the sum of the intensity footprints of the selected
        masks in each plane. Masks are included in the sum if they are selected by the
        user (i.e. True in idx_selected).

        Returns
        -------
        mask_image_by_plane : np.ndarray (float)
            The mask image for each plane in the volume. Filters the masks in each plane
            by idx_selected and sums over their intensity footprints to create a
            single image for each plane.
        """
        idx_selected = self.idx_selected
        image_data = np.zeros(
            (
                self.roi_processor.num_planes,
                self.roi_processor.ly,
                self.roi_processor.lx,
            ),
            dtype=float,
        )
        for iroi, (plane, lam, ypix, xpix) in enumerate(
            zip(
                self.roi_processor.plane_idx,
                self.roi_processor.lam,
                self.roi_processor.ypix,
                self.roi_processor.xpix,
            )
        ):
            if idx_selected[iroi]:
                image_data[plane, ypix, xpix] += lam
        return image_data

    @property
    def mask_labels(self):
        """Return the masks in each volume as labels by summing the selected masks in each plane.

        ROIs are assigned an index that is unique across all ROIs independent of plane.
        The index is offset by 1 because napari uses 0 to indicate "no label". ROIs are
        only presented if the are currently "selected" by the user (i.e. True in
        idx_selected).

        Returns
        -------
        mask_labels_by_plane : np.ndarray (int)
            Each pixel is assigned a label associated with each ROI. The label is the
            index to the ROI - and if ROIs are overlapping then the last ROI will be
            used. Only ROIs that are selected by the user are included in the labels.
        """
        idx_selected = self.idx_selected
        label_data = np.zeros(
            (
                self.roi_processor.num_planes,
                self.roi_processor.ly,
                self.roi_processor.lx,
            ),
            dtype=int,
        )
        for iroi, (plane, ypix, xpix) in enumerate(
            zip(
                self.roi_processor.plane_idx,
                self.roi_processor.ypix,
                self.roi_processor.xpix,
            )
        ):
            if idx_selected[iroi]:
                label_data[plane, ypix, xpix] = iroi + 1
        return label_data

    @property
    def idx_selected(self):
        """Return a boolean index of the currently selected masks.

        Returns
        -------
        idx_selected_by_plane : np.ndarray (bool)
            The indices of the selected masks across planes.
        """
        if self.only_manual_labels:
            # if only manual labels, ignore the feature criteria
            idx = np.full(self.roi_processor.num_rois, False)
        else:
            # otherwise, use the feature criteria
            if self.show_control_cells:
                idx = np.copy(~self.idx_meets_criteria)
            else:
                idx = np.copy(self.idx_meets_criteria)
        if self.use_manual_labels:
            idx[self.manual_label_active] = (
                self.manual_label[self.manual_label_active] != self.show_control_cells
            )
        return idx

    @property
    def idx_target(self):
        """Return a boolean index of the target masks.

        Any mask that is True in this index is considered a target mask. These are all the masks
        that meet feature criteria and have manual labels.

        Returns
        -------
        idx_target : np.ndarray (bool)
            The indices of the target masks across planes.
        """
        # these meet feature criteria for active features
        idx_target = np.copy(self.idx_meets_criteria)
        # overwrite any that have manual labels
        idx_target[self.manual_label_active] = self.manual_label[
            self.manual_label_active
        ]
        return idx_target

    # ------------------------------------------------------------------------------------------------
    # ----------------------------------- GUI Initialization -----------------------------------------
    # ------------------------------------------------------------------------------------------------
    def _initialize_napari_viewer(self):
        """Initialize the napari viewer and associated GUI features.

        This function initializes the napari viewer and adds the reference image, masks,
        and labels to the viewer. It also creates the GUI features for the histograms of
        the intensity features of the selected cells and adds them to the viewer. The GUI features
        include toggle buttons for selecting the range of feature values that qualify as selected
        cells, buttons for saving the selected cell selection and toggling between control and selected
        cells, and buttons for toggling the visibility of the masks and labels.

        There are additional key stroke controls for efficient control of the GUI.
        """
        self.viewer = napari.Viewer(title=f"Cell Curation")
        self.reference = self.viewer.add_image(
            np.stack(self.roi_processor.references),
            name="reference",
            blending="additive",
            opacity=0.6,
        )
        self.masks = self.viewer.add_image(
            self.mask_image,
            name="masks_image",
            blending="additive",
            colormap="red",
            visible=self.show_mask_image,
        )
        self.labels = self.viewer.add_labels(
            self.mask_labels,
            name="mask_labels",
            blending="additive",
            visible=not self.show_mask_image,
        )
        self.viewer.dims.current_step = (
            self.plane_idx,
            self.viewer.dims.current_step[1],
            self.viewer.dims.current_step[2],
        )

        # Build feature window area to add to the napari viewer
        self.feature_window = pg.GraphicsLayoutWidget()
        self.text_area = pg.LabelItem(
            "welcome to the cell selector GUI", justify="left"
        )
        self.toggle_area = pg.GraphicsLayout()
        self.plot_area = pg.GraphicsLayout()
        self.button_area = pg.GraphicsLayout()
        self.feature_window.addItem(self.text_area, row=0, col=0)
        self.feature_window.addItem(self.toggle_area, row=1, col=0)
        self.feature_window.addItem(self.plot_area, row=2, col=0)
        self.feature_window.addItem(self.button_area, row=3, col=0)

        # Build elements of the feature window area
        self._prepare_feature_histograms()
        self._build_histograms()
        self._build_cutoff_lines()
        # Reset selection based on feature cutoffs if saved cutoffs were found
        self.update_by_feature_criterion()
        self._build_feature_toggles()
        self._build_buttons()

        # Add the feature window to the napari viewer
        self.dock_window = self.viewer.window.add_dock_widget(
            self.feature_window, name="ROI Features", area="bottom"
        )

        self.viewer.bind_key("t", self._toggle_cells_to_view, overwrite=True)
        self.viewer.bind_key("s", self._switch_image_label, overwrite=True)
        self.viewer.bind_key("v", self._update_mask_visibility, overwrite=True)
        self.viewer.bind_key("r", self._update_reference_visibility, overwrite=True)
        self.viewer.bind_key("c", self._next_color_state, overwrite=True)
        self.viewer.bind_key("a", self._next_colormap, overwrite=True)

        self.labels.mouse_drag_callbacks.append(self._single_click_label)
        self.masks.mouse_drag_callbacks.append(self._single_click_label)
        self.reference.mouse_drag_callbacks.append(self._single_click_label)
        self.labels.mouse_double_click_callbacks.append(self._double_click_label)
        self.masks.mouse_double_click_callbacks.append(self._double_click_label)
        self.reference.mouse_double_click_callbacks.append(self._double_click_label)

        self.viewer.dims.events.connect(self._update_plane_idx)

    # ------------------------------------------------------------------------------------------------
    # ---------------------------------------- GUI Components ----------------------------------------
    # ------------------------------------------------------------------------------------------------
    def _prepare_feature_histograms(self):
        """Prepare the histograms for the intensity features of the target cells.

        This function computes the histograms of the intensity features of all the cells
        and stores the histograms in a format that can be used to update the histograms
        in the GUI. The histograms are computed for each plane separately, and the maximum
        value for the y-range of the histograms is set independently for each feature which
        constrains the users scrolling to a useful range.
        """
        self.h_values_full = {
            key: [None] * self.roi_processor.num_planes
            for key in self.roi_processor.features
        }
        self.h_values_selected = {
            key: [None] * self.roi_processor.num_planes
            for key in self.roi_processor.features
        }
        self.h_bin_edges = {key: [None] for key in self.roi_processor.features}

        # set the edges of the histograms for each feature (this is the same across planes)
        for feature_name, feature_values in self.roi_processor.features.items():
            feature_edges = np.histogram(feature_values, bins=self.num_bins)[1]
            self.h_bin_edges[feature_name] = feature_edges

        # compute histograms for each feature in each plane
        features_by_plane = {
            key: split_planes(value, self.roi_processor.rois_per_plane)
            for key, value in self.roi_processor.features.items()
        }
        idx_selected_by_plane = split_planes(
            self.idx_selected, self.roi_processor.rois_per_plane
        )
        for feature in self.roi_processor.features:
            for iplane in range(self.roi_processor.num_planes):
                all_values_this_plane = features_by_plane[feature][iplane]
                sel_values_this_plane = all_values_this_plane[
                    idx_selected_by_plane[iplane]
                ]
                self.h_values_full[feature][iplane] = np.histogram(
                    all_values_this_plane, bins=self.h_bin_edges[feature]
                )[0]
                self.h_values_selected[feature][iplane] = np.histogram(
                    sel_values_this_plane, bins=self.h_bin_edges[feature]
                )[0]

        # set the maximum value for the y-range of the histograms independently for each feature
        self.h_values_maximum = {
            key: max(np.concatenate(value)) for key, value in self.h_values_full.items()
        }

    def _build_histograms(self):
        """Build the histograms of the intensity features of the target cells in the GUI."""
        self.hist_layout = pg.GraphicsLayout()
        self.hist_graphs = {key: [None] for key in self.roi_processor.features}
        self.hist_selected = {key: [None] for key in self.roi_processor.features}
        for feature in self.roi_processor.features:
            bar_width = np.diff(self.h_bin_edges[feature][:2])
            bin_centers = self.h_bin_edges[feature][:-1] + bar_width / 2
            height_full = self.h_values_full[feature][self.plane_idx]
            height_selected = self.h_values_selected[feature][self.plane_idx]
            self.hist_graphs[feature] = pg.BarGraphItem(
                x=bin_centers, height=height_full, width=bar_width
            )
            self.hist_selected[feature] = pg.BarGraphItem(
                x=bin_centers, height=height_selected, width=bar_width, brush="r"
            )

        self.preserve_methods = {}

        self.hist_plots = {key: [None] for key in self.roi_processor.features}
        for ifeature, feature in enumerate(self.roi_processor.features):
            self.hist_plots[feature] = self.plot_area.addPlot(
                row=0, col=ifeature, title=feature
            )
            self.hist_plots[feature].setMouseEnabled(x=False)
            self.hist_plots[feature].setYRange(0, self.h_values_maximum[feature])
            self.hist_plots[feature].addItem(self.hist_graphs[feature])
            self.hist_plots[feature].addItem(self.hist_selected[feature])
            self.preserve_methods[feature] = functools.partial(
                self._preserve_y_range, feature=feature
            )
            self.hist_plots[feature].getViewBox().sigYRangeChanged.connect(
                self.preserve_methods[feature]
            )

    def _build_cutoff_lines(self):
        self.feature_range = {}
        self.feature_cutoffs = {}
        self.cutoff_lines = {}
        for feature in self.roi_processor.features:
            self.feature_range[feature] = [
                np.min(self.h_bin_edges[feature]),
                np.max(self.h_bin_edges[feature]),
            ]

            # Try loading feature cutoffs
            load_successful = False
            if io.is_criteria_saved(self.roi_processor.root_dir, feature):
                cutoffs = io.load_criteria(self.roi_processor.root_dir, feature)
                if cutoffs[0] is None:
                    cutoffs[0] = self.feature_range[feature][0]
                    self.feature_active[feature][0] = False
                if cutoffs[1] is None:
                    cutoffs[1] = self.feature_range[feature][1]
                    self.feature_active[feature][1] = False
                # make sure cutoffs are in order from lowest to highest
                cutoffs = sorted(cutoffs)
                self.feature_cutoffs[feature] = cutoffs
                load_successful = True
            if not load_successful:
                # if loading fails, then set the cutoffs to the full range
                self.feature_cutoffs[feature] = copy(self.feature_range[feature])
            self.cutoff_lines[feature] = [None] * 2
            for i in range(2):
                if self.feature_active[feature][i]:
                    self.cutoff_lines[feature][i] = pg.InfiniteLine(
                        pos=self.feature_cutoffs[feature][i], movable=True
                    )
                else:
                    self.cutoff_lines[feature][i] = pg.InfiniteLine(
                        pos=self.feature_range[feature][i], movable=False
                    )
                self.cutoff_lines[feature][i].setBounds(self.feature_range[feature])
                self.cutoff_lines[feature][i].sigPositionChangeFinished.connect(
                    functools.partial(self._update_cutoff_finished, feature=feature)
                )
                self.hist_plots[feature].addItem(self.cutoff_lines[feature][i])

    def _build_feature_toggles(self):
        self.min_max_name = ["min", "max"]
        self.max_length_name = (
            max([len(feature) for feature in self.roi_processor.features]) + 9
        )

        self.use_feature_buttons = {
            key: [None, None] for key in self.roi_processor.features
        }
        self.use_feature_proxies = [None] * (self.num_features * 2)
        for ifeature, feature in enumerate(self.roi_processor.features):
            for i in range(2):
                proxy_idx = 2 * ifeature + i
                if self.feature_active[feature][i]:
                    text_to_use = f"using {self.min_max_name[i]} {feature}".center(
                        self.max_length_name, " "
                    )
                    style_to_use = q_not_checked_style
                else:
                    text_to_use = f"ignore {self.min_max_name[i]} {feature}".center(
                        self.max_length_name, " "
                    )
                    style_to_use = q_checked_style
                self.use_feature_buttons[feature][i] = QPushButton(
                    "toggle", text=text_to_use
                )
                self.use_feature_buttons[feature][i].setCheckable(True)
                self.use_feature_buttons[feature][i].setChecked(
                    self.feature_active[feature][i]
                )
                self.use_feature_buttons[feature][i].clicked.connect(
                    functools.partial(self._toggle_feature, feature=feature, iminmax=i)
                )
                self.use_feature_buttons[feature][i].setStyleSheet(style_to_use)
                self.use_feature_proxies[proxy_idx] = QGraphicsProxyWidget()
                self.use_feature_proxies[proxy_idx].setWidget(
                    self.use_feature_buttons[feature][i]
                )
                self.toggle_area.addItem(
                    self.use_feature_proxies[proxy_idx], row=0, col=proxy_idx
                )

    def _build_buttons(self):
        self.save_button = QPushButton("button", text="save selection")
        self.save_button.clicked.connect(self.save_selection)
        self.save_button.setStyleSheet(basic_button_style)
        self.save_proxy = QGraphicsProxyWidget()
        self.save_proxy.setWidget(self.save_button)

        self.toggle_cell_button = QPushButton(
            text="control cells" if self.show_control_cells else "target cells"
        )
        self.toggle_cell_button.clicked.connect(self._toggle_cells_to_view)
        self.toggle_cell_button.setStyleSheet(basic_button_style)
        self.toggle_cell_proxy = QGraphicsProxyWidget()
        self.toggle_cell_proxy.setWidget(self.toggle_cell_button)

        self.use_manual_labels_button = QPushButton(
            text=(
                "using manual labels"
                if self.use_manual_labels
                else "ignoring manual labels"
            )
        )
        self.use_manual_labels_button.clicked.connect(self._toggle_use_manual_labels)
        self.use_manual_labels_button.setStyleSheet(basic_button_style)
        self.use_manual_labels_proxy = QGraphicsProxyWidget()
        self.use_manual_labels_proxy.setWidget(self.use_manual_labels_button)

        self.clear_manual_label_button = QPushButton(text="clear manual labels")
        self.clear_manual_label_button.clicked.connect(self._clear_manual_labels)
        self.clear_manual_label_button.setStyleSheet(basic_button_style)
        self.clear_manual_label_proxy = QGraphicsProxyWidget()
        self.clear_manual_label_proxy.setWidget(self.clear_manual_label_button)

        self.show_manual_labels_button = QPushButton(text="all labels")
        self.show_manual_labels_button.clicked.connect(self._show_manual_labels)
        self.show_manual_labels_button.setStyleSheet(basic_button_style)
        self.show_manual_labels_proxy = QGraphicsProxyWidget()
        self.show_manual_labels_proxy.setWidget(self.show_manual_labels_button)

        self.color_button = QPushButton(text=self.color_state_names[self.color_state])
        self.color_button.setCheckable(False)
        self.color_button.clicked.connect(self._next_color_state)
        self.color_button.setStyleSheet(basic_button_style)
        self.color_proxy = QGraphicsProxyWidget()
        self.color_proxy.setWidget(self.color_button)

        self.colormap_selection = QPushButton(text=self.colormaps[self.idx_colormap])
        self.colormap_selection.clicked.connect(self._next_colormap)
        self.colormap_selection.setStyleSheet(basic_button_style)
        self.colormap_proxy = QGraphicsProxyWidget()
        self.colormap_proxy.setWidget(self.colormap_selection)

        self.button_area.addItem(self.save_proxy, row=0, col=0)
        self.button_area.addItem(self.toggle_cell_proxy, row=0, col=1)
        self.button_area.addItem(self.use_manual_labels_proxy, row=0, col=2)
        self.button_area.addItem(self.show_manual_labels_proxy, row=0, col=3)
        self.button_area.addItem(self.clear_manual_label_proxy, row=0, col=4)
        self.button_area.addItem(self.color_proxy, row=0, col=5)
        self.button_area.addItem(self.colormap_proxy, row=0, col=6)

    # ------------------------------------------------------------------------------------------------
    # ---------------------------------- Callbacks for GUI elements ----------------------------------
    # ------------------------------------------------------------------------------------------------
    def _preserve_y_range(self, feature):
        """Support for preserving the y limits of the feature histograms in a useful range."""
        # remove callback so we can update the yrange without a recursive call
        self.hist_plots[feature].getViewBox().sigYRangeChanged.disconnect(
            self.preserve_methods[feature]
        )
        # then figure out the current y range (this is after a user update)
        current_min, current_max = self.hist_plots[feature].viewRange()[1]
        # set the new max to not exceed the current maximum
        current_range = current_max - current_min
        current_max = min(current_range, self.h_values_maximum[feature])
        # range is from 0 to the max, therefore the y=0 line always stays in the same place
        self.hist_plots[feature].setYRange(0, current_max)
        # reconnect callback for next update
        self.hist_plots[feature].getViewBox().sigYRangeChanged.connect(
            self.preserve_methods[feature]
        )

    def _update_cutoff_finished(self, event, feature):
        """Callback for updating the feature cutoffs when the user finishes moving the cutoff lines."""
        cutoff_values = [
            self.cutoff_lines[feature][0].pos()[0],
            self.cutoff_lines[feature][1].pos()[0],
        ]
        # store cutoffs from minimum to maximum
        min_cutoff, max_cutoff = min(cutoff_values), max(cutoff_values)
        self.feature_cutoffs[feature][0] = min_cutoff
        self.feature_cutoffs[feature][1] = max_cutoff
        self.cutoff_lines[feature][0].setValue(min_cutoff)
        self.cutoff_lines[feature][1].setValue(max_cutoff)
        self.update_by_feature_criterion()

    def _toggle_use_manual_labels(self, event):
        self.use_manual_labels = not self.use_manual_labels
        self.use_manual_labels_button.setText(
            "using manual labels"
            if self.use_manual_labels
            else "ignoring manual labels"
        )
        self.regenerate_mask_data()  # update replot masks and recompute histograms
        self.update_text(
            f"{'using' if self.use_manual_labels else 'ignoring'} manual labels"
        )

    def _show_manual_labels(self, event):
        self.only_manual_labels = not self.only_manual_labels
        if self.only_manual_labels:
            self.use_manual_labels = True
        self.show_manual_labels_button.setText(
            "only manual labels" if self.only_manual_labels else "all labels"
        )
        self.regenerate_mask_data()
        self.update_text(
            "only showing manual labels"
            if self.only_manual_labels
            else "showing all labels"
        )

    def _clear_manual_labels(self, event):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ControlModifier:
            self.manual_label_active[:] = False
            self.regenerate_mask_data()
            self.update_text("You just cleared all manual labels!")
        else:
            self.update_text(
                "Clearing manual labels requires a control click for safety! Try again."
            )

    def _toggle_feature(self, event, feature, iminmax):
        self.feature_active[feature][iminmax] = self.use_feature_buttons[feature][
            iminmax
        ].isChecked()
        if self.feature_active[feature][iminmax]:
            text_to_use = f"using {self.min_max_name[iminmax]} {feature}".center(
                self.max_length_name, " "
            )
            self.cutoff_lines[feature][iminmax].setValue(
                self.feature_cutoffs[feature][iminmax]
            )
            self.cutoff_lines[feature][iminmax].setMovable(True)
            self.use_feature_buttons[feature][iminmax].setText(text_to_use)
            self.use_feature_buttons[feature][iminmax].setStyleSheet(
                q_not_checked_style
            )
        else:
            text_to_use = f"ignore {self.min_max_name[iminmax]} {feature}".center(
                self.max_length_name, " "
            )
            self.cutoff_lines[feature][iminmax].setValue(
                self.feature_range[feature][iminmax]
            )
            self.cutoff_lines[feature][iminmax].setMovable(False)
            self.use_feature_buttons[feature][iminmax].setText(text_to_use)
            self.use_feature_buttons[feature][iminmax].setStyleSheet(q_checked_style)

        # update selection, which will replot everything
        self.update_by_feature_criterion()

    def _toggle_cells_to_view(self, event):
        # changes whether to plot control or target cells (maybe add a textbox and update it so as to not depend on looking at the print outputs...)
        self.show_control_cells = not self.show_control_cells
        self.toggle_cell_button.setText(
            "control cells" if self.show_control_cells else "target cells"
        )
        self.masks.data = self.mask_image
        self.labels.data = self.mask_labels
        self.regenerate_mask_data()
        self.update_text(
            f"Now viewing {'control' if self.show_control_cells else 'target'} cells"
        )

    def _next_color_state(self, event):
        self.color_state = np.mod(self.color_state + 1, len(self.color_state_names))
        self.color_button.setText(self.color_state_names[self.color_state])
        self.update_label_colors()
        self.update_text(f"now coloring by {self.color_state_names[self.color_state]}")

    def _next_colormap(self, event):
        self.idx_colormap = np.mod(self.idx_colormap + 1, len(self.colormaps))
        self.colormap_selection.setText(self.colormaps[self.idx_colormap])
        self.update_label_colors()

    def _switch_image_label(self, event):
        self.show_mask_image = not self.show_mask_image
        self.update_visibility()
        self.update_text(
            f"now showing {'mask image' if self.show_mask_image else 'mask labels'}"
        )

    def _update_mask_visibility(self, event):
        self.mask_visibility = not self.mask_visibility
        self.update_visibility()

    def _update_reference_visibility(self, event):
        self.reference.visible = not self.reference.visible

    # create single-click callback for printing data about ROI features
    def _single_click_label(self, _, event):
        if not self.labels.visible:
            self.update_text(
                "can only manually select cells when the labels are visible!"
            )
            return

        # get click data
        plane_idx, yidx, xidx = [int(pos) for pos in event.position]
        label_idx = self.labels.data[plane_idx, yidx, xidx]
        if label_idx == 0:
            self.update_text("single-click on background, no ROI selected")
            return

        # get ROI data
        roi_idx = label_idx - 1  # oh napari, oh napari
        feature_print = [
            f"{feature}={fvalue[roi_idx]:.3f}"
            for feature, fvalue in self.roi_processor.features.items()
        ]

        string_to_print = f"ROI: {roi_idx}" + " ".join(feature_print)

        # only print single click data if alt is held down
        if "Alt" in event.modifiers:
            print(string_to_print)

        # always show message in viewer status
        self.update_text(string_to_print)

    def _double_click_label(self, _, event):
        self.update_text(
            "you just double clicked!"
        )  # will be overwritten - useful for debugging

        # if not looking at labels, then don't allow manual selection (it would be random!)
        if not self.labels.visible:
            self.update_text(
                "can only manually select cells when the labels are visible!"
            )
            return

        # if not looking at manual annotations, don't allow manual selection...
        if not self.use_manual_labels:
            self.update_text(
                "can only manually select cells when the manual labels are being used!"
            )
            return

        plane_idx, yidx, xidx = [int(pos) for pos in event.position]
        label_idx = self.labels.data[plane_idx, yidx, xidx]
        if label_idx == 0:
            self.update_text("double-click on background, no ROI selected")
        else:
            if "Alt" in event.modifiers:
                self.update_text(
                    "Alt was used, assuming you are trying to single click and not doing a manual label!"
                )
            else:
                roi_idx = label_idx - 1
                if "Control" in event.modifiers:
                    if self.only_manual_labels:
                        self.manual_label_active[roi_idx] = False
                        self.update_text(
                            f"you just removed the manual label from roi: {roi_idx}"
                        )
                    else:
                        self.update_text(
                            f"you can only remove a label if you are only looking at manual labels!"
                        )
                else:
                    # manual annotation: if plotting control cells, then annotate as target (1), if plotting target cells, annotate as control (0)
                    new_label = copy(self.show_control_cells)
                    self.manual_label[roi_idx] = new_label
                    self.manual_label_active[roi_idx] = True
                    self.update_text(
                        f"you just labeled roi: {roi_idx} with the identity: {new_label}"
                    )
                self.regenerate_mask_data()

    def _update_plane_idx(self, event):
        """Callback for the dimension slider to coordinate feature histograms with the viewer."""
        self.plane_idx = event.source.current_step[0]
        self.update_feature_plots()
