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

from .red_cell_processor import RedCellProcessor
from . import utils

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


class RedSelectionGUI:
    """A GUI for selecting red cells from a suite2p session.

    This GUI allows the user to interactively select red cells from a suite2p session.
    The user can select red cells based on the intensity features of the cells. The GUI
    shows histograms of the intensity features of the red cells and allows the user to
    select the range of feature values that qualify as red cells. The GUI also allows
    the user to manually label cells as red or control cells.

    Parameters
    ----------
    rcp : RedCellProcessor
        An instance of the RedCellProcessor class that contains the suite2p session data.
    num_bins : int, optional
        The number of bins to use for the histograms of the intensity features. Default is 50.
    """

    def __init__(self, rcp, num_bins=50):
        if not isinstance(rcp, RedCellProcessor):
            raise ValueError("redCellObj must be an instance of the redCellProcessing class inherited from session")

        self.rcp = rcp
        self.num_bins = num_bins
        self.plane_idx = 0  # determines which plane is currently being shown in the napari viewer

        self.num_features = len(self.rcp.features)
        self.feature_active = {key: [True, True] for key in self.rcp.features}

        # process initial plane
        self.red_idx = np.full(self.rcp.num_rois, True)
        self.manual_label = np.full(self.rcp.num_rois, False)
        self.manual_label_active = np.full(self.rcp.num_rois, False)
        self._prepare_feature_histograms()

        # open napari viewer and associated GUI features
        self.show_control_cells = False  # show control cells instead of red cells
        self.show_mask_image = False  # if true, will show mask image, if false, will show mask labels
        self.mask_visibility = True  # if true, will show either mask image or label, otherwise will not show either!
        self.use_manual_labels = True  # if true, then will apply manual labels after using features to compute red_idx
        self.only_manual_labels = False  # if true, only show manual labels of selected category...
        self.color_state = 0  # indicates which color to display maskLabels (0:random, 1-4:color by feature)
        self.color_state_names = ["random", *self.rcp.features.keys()]
        self.idx_colormap = 0  # which colormap to use for pseudo coloring the masks
        self.colormaps = ["plasma", "autumn", "spring", "summer", "winter", "hot"]
        self._initialize_napari_viewer()

    def _prepare_feature_histograms(self):
        """Prepare the histograms for the intensity features of the red cells.

        This function computes the histograms of the intensity features of the red cells
        and stores the histograms in a format that can be used to update the histograms
        in the GUI. The histograms are computed for each plane separately, and the maximum
        value for the y-range of the histograms is set independently for each feature which
        constrains the users scrolling to a useful range.
        """
        self.h_values = {key: [None] * self.rcp.num_planes for key in self.rcp.features}
        self.h_values_red = {key: [None] * self.rcp.num_planes for key in self.rcp.features}
        self.h_bin_edges = {key: [None] for key in self.rcp.features}

        # set the edges of the histograms for each feature (this is the same across planes)
        for feature_name, feature_values in self.rcp.features.items():
            feature_edges = np.histogram(feature_values, bins=self.num_bins)[1]
            self.h_bin_edges[feature_name] = feature_edges

        # compute histograms for each feature in each plane
        features_by_plane = {key: utils.split_planes(value, self.rcp.rois_per_plane) for key, value in self.rcp.features.items()}
        idx_selected_by_plane = utils.split_planes(self.red_idx, self.rcp.rois_per_plane)
        for feature in self.rcp.features:
            for iplane in range(self.rcp.num_planes):
                all_values_this_plane = features_by_plane[feature][iplane]
                red_values_this_plane = all_values_this_plane[idx_selected_by_plane[iplane]]
                self.h_values[feature][iplane] = np.histogram(all_values_this_plane, bins=self.h_bin_edges[feature])[0]
                self.h_values_red[feature][iplane] = np.histogram(red_values_this_plane, bins=self.h_bin_edges[feature])[0]

        # set the maximum value for the y-range of the histograms independently for each feature
        self.h_values_maximum = {key: max(np.concatenate(value)) for key, value in self.h_values.items()}

    def _initialize_napari_viewer(self):
        """Initialize the napari viewer and associated GUI features.

        This function initializes the napari viewer and adds the reference image, masks,
        and labels to the viewer. It also creates the GUI features for the histograms of
        the intensity features of the red cells and adds them to the viewer. The GUI features
        include toggle buttons for selecting the range of feature values that qualify as red
        cells, buttons for saving the red cell selection and toggling between control and red
        cells, and buttons for toggling the visibility of the masks and labels.

        There are additional key stroke controls for efficient control of the GUI.
        """
        self.viewer = napari.Viewer(title=f"Red Cell Curation")
        self.reference = self.viewer.add_image(np.stack(self.rcp.references), name="reference", blending="additive", opacity=0.6)
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

        self.feature_window = pg.GraphicsLayoutWidget()

        self.toggle_area = pg.GraphicsLayout()
        self.plot_area = pg.GraphicsLayout()
        self.button_area = pg.GraphicsLayout()
        self.feature_window.addItem(self.toggle_area, row=0, col=0)
        self.feature_window.addItem(self.plot_area, row=1, col=0)
        self.feature_window.addItem(self.button_area, row=2, col=0)

        self.hist_layout = pg.GraphicsLayout()
        self.hist_graphs = {key: [None] for key in self.rcp.features}
        self.hist_reds = {key: [None] for key in self.rcp.features}
        for feature in self.rcp.features:
            bar_width = np.diff(self.h_bin_edges[feature][:2])
            bin_centers = self.h_bin_edges[feature][:-1] + bar_width / 2
            self.hist_graphs[feature] = pg.BarGraphItem(x=bin_centers, height=self.h_values[feature][self.plane_idx], width=bar_width)
            self.hist_reds[feature] = pg.BarGraphItem(x=bin_centers, height=self.h_values_red[feature][self.plane_idx], width=bar_width, brush="r")

        self.preserve_methods = {}

        def preserve_y_range(feature):
            """Support for preserving the y limits of the feature histograms in a useful range."""
            # remove callback so we can update the yrange without a recursive call
            self.hist_plots[feature].getViewBox().sigYRangeChanged.disconnect(self.preserve_methods[feature])
            # then figure out the current y range (this is after a user update)
            current_min, current_max = self.hist_plots[feature].viewRange()[1]
            # set the new max to not exceed the current maximum
            current_range = current_max - current_min
            current_max = min(current_range, self.h_values_maximum[feature])
            # range is from 0 to the max, therefore the y=0 line always stays in the same place
            self.hist_plots[feature].setYRange(0, current_max)
            # reconnect callback for next update
            self.hist_plots[feature].getViewBox().sigYRangeChanged.connect(self.preserve_methods[feature])

        # add bargraphs to plotArea
        self.hist_plots = {key: [None] for key in self.rcp.features}
        for ifeature, feature in enumerate(self.rcp.features):
            self.hist_plots[feature] = self.plot_area.addPlot(row=0, col=ifeature, title=feature)
            self.hist_plots[feature].setMouseEnabled(x=False)
            self.hist_plots[feature].setYRange(0, self.h_values_maximum[feature])
            self.hist_plots[feature].addItem(self.hist_graphs[feature])
            self.hist_plots[feature].addItem(self.hist_reds[feature])
            self.preserve_methods[feature] = functools.partial(preserve_y_range, feature=feature)
            self.hist_plots[feature].getViewBox().sigYRangeChanged.connect(self.preserve_methods[feature])

        # create cutoffLines (vertical infinite lines) for determining the range within feature values that qualify as red
        def update_cutoff_finished(_, feature):
            """Callback for updating the feature cutoffs when the user finishes moving the cutoff lines."""
            cutoff_values = [
                self.cutoff_lines[feature][0].pos()[0],
                self.cutoff_lines[feature][1].pos()[0],
            ]
            min_cutoff, max_cutoff = min(cutoff_values), max(cutoff_values)  # order of cutoff lines doesn't matter
            self.feature_cutoffs[feature][0] = min_cutoff
            self.feature_cutoffs[feature][1] = max_cutoff
            self.cutoff_lines[feature][0].setValue(min_cutoff)
            self.cutoff_lines[feature][1].setValue(max_cutoff)
            self.update_by_feature_criterion()

        self.feature_range = {}
        self.feature_cutoffs = {}
        self.cutoff_lines = {}
        for feature in self.rcp.features:
            self.feature_range[feature] = [np.min(self.h_bin_edges[feature]), np.max(self.h_bin_edges[feature])]
            # TODO: Load from previous if available
            self.feature_cutoffs[feature] = copy(self.feature_range[feature])
            self.cutoff_lines[feature] = [None] * 2
            for i in range(2):
                if self.feature_active[feature][i]:
                    self.cutoff_lines[feature][i] = pg.InfiniteLine(pos=self.feature_cutoffs[feature][i], movable=True)
                else:
                    self.cutoff_lines[feature][i] = pg.InfiniteLine(pos=self.feature_range[feature][i], movable=False)
                self.cutoff_lines[feature][i].setBounds(self.feature_range[feature])
                self.cutoff_lines[feature][i].sigPositionChangeFinished.connect(functools.partial(update_cutoff_finished, feature=feature))
                self.hist_plots[feature].addItem(self.cutoff_lines[feature][i])

        # Reset red selection
        self.update_by_feature_criterion()

        # ---------------------
        # -- now add toggles --
        # ---------------------
        min_max_name = ["min", "max"]
        max_length_name = max([len(feature) for feature in self.rcp.features]) + 9

        def toggle_feature(_, feature, iminmax):
            self.feature_active[feature][iminmax] = self.use_feature_buttons[feature][iminmax].isChecked()
            if self.feature_active[feature][iminmax]:
                text_to_use = f"using {min_max_name[iminmax]} {feature}".center(max_length_name, " ")
                self.cutoff_lines[feature][iminmax].setValue(self.feature_cutoffs[feature][iminmax])
                self.cutoff_lines[feature][iminmax].setMovable(True)
                self.use_feature_buttons[feature][iminmax].setText(text_to_use)
                self.use_feature_buttons[feature][iminmax].setStyleSheet(q_not_checked_style)
            else:
                text_to_use = f"ignore {min_max_name[iminmax]} {feature}".center(max_length_name, " ")
                self.cutoff_lines[feature][iminmax].setValue(self.feature_range[feature][iminmax])
                self.cutoff_lines[feature][iminmax].setMovable(False)
                self.use_feature_buttons[feature][iminmax].setText(text_to_use)
                self.use_feature_buttons[feature][iminmax].setStyleSheet(q_checked_style)

            # update red selection, which will replot everything
            self.update_by_feature_criterion()

        self.use_feature_buttons = {key: [None, None] for key in self.rcp.features}
        self.use_feature_proxies = [None] * (self.num_features * 2)
        for ifeature, feature in enumerate(self.rcp.features):
            for i in range(2):
                proxy_idx = 2 * ifeature + i
                if self.feature_active[feature][i]:
                    text_to_use = f"using {min_max_name[i]} {feature}".center(max_length_name, " ")
                    style_to_use = q_not_checked_style
                else:
                    text_to_use = f"ignore {min_max_name[i]} {feature}".center(max_length_name, " ")
                    style_to_use = q_checked_style
                self.use_feature_buttons[feature][i] = QPushButton("toggle", text=text_to_use)
                self.use_feature_buttons[feature][i].setCheckable(True)
                self.use_feature_buttons[feature][i].setChecked(self.feature_active[feature][i])
                self.use_feature_buttons[feature][i].clicked.connect(functools.partial(toggle_feature, feature=feature, iminmax=i))
                self.use_feature_buttons[feature][i].setStyleSheet(style_to_use)
                self.use_feature_proxies[proxy_idx] = QGraphicsProxyWidget()
                self.use_feature_proxies[proxy_idx].setWidget(self.use_feature_buttons[feature][i])
                self.toggle_area.addItem(self.use_feature_proxies[proxy_idx], row=0, col=proxy_idx)

        # ---------------------
        # -- now add buttons --
        # ---------------------

        def save_rois(_):
            self.save_selection()

        self.save_button = QPushButton("button", text="save red selection")
        self.save_button.clicked.connect(save_rois)
        self.save_button.setStyleSheet(basic_button_style)
        self.save_proxy = QGraphicsProxyWidget()
        self.save_proxy.setWidget(self.save_button)

        # add toggle control/red cell button
        def toggle_cells_to_view(_):
            # changes whether to plot control or red cells (maybe add a textbox and update it so as to not depend on looking at the print outputs...)
            self.show_control_cells = not self.show_control_cells
            self.toggle_cell_button.setText("control cells" if self.show_control_cells else "red cells")
            self.masks.data = self.mask_image
            self.labels.data = self.mask_labels

        self.toggle_cell_button = QPushButton(text="control cells" if self.show_control_cells else "red cells")
        self.toggle_cell_button.clicked.connect(toggle_cells_to_view)
        self.toggle_cell_button.setStyleSheet(basic_button_style)
        self.toggle_cell_proxy = QGraphicsProxyWidget()
        self.toggle_cell_proxy.setWidget(self.toggle_cell_button)

        # add button to toggle whether to include manual labels in mask plot
        def toggle_use_manual_labels(_):
            self.use_manual_labels = not self.use_manual_labels
            self.use_manual_labels_button.setText("using manual labels" if self.use_manual_labels else "ignoring manual labels")
            # update replot masks and recompute histograms
            self.regenerate_mask_data()

        self.use_manual_labels_button = QPushButton(text="using manual labels" if self.use_manual_labels else "ignoring manual labels")
        self.use_manual_labels_button.clicked.connect(toggle_use_manual_labels)
        self.use_manual_labels_button.setStyleSheet(basic_button_style)
        self.use_manual_labels_proxy = QGraphicsProxyWidget()
        self.use_manual_labels_proxy.setWidget(self.use_manual_labels_button)

        # add button to clear all manual labels
        def clear_manual_labels(_):
            modifiers = QtWidgets.QApplication.keyboardModifiers()
            if modifiers == QtCore.Qt.ControlModifier:
                self.manual_label_active[:] = False
                self.regenerate_mask_data()
            else:
                print("clearing manual labels requires a control click")

        self.clear_manual_label_button = QPushButton(text="clear manual labels")
        self.clear_manual_label_button.clicked.connect(clear_manual_labels)
        self.clear_manual_label_button.setStyleSheet(basic_button_style)
        self.clear_manual_label_proxy = QGraphicsProxyWidget()
        self.clear_manual_label_proxy.setWidget(self.clear_manual_label_button)

        # add show manual labels only button
        def show_manual_labels(_):
            self.only_manual_labels = not self.only_manual_labels
            if self.only_manual_labels:
                self.use_manual_labels = True
            self.show_manual_labels_button.setText("only manual labels" if self.onlyManualLabels else "all labels")
            self.regenerate_mask_data()

        self.show_manual_labels_button = QPushButton(text="all labels")
        self.show_manual_labels_button.clicked.connect(show_manual_labels)
        self.show_manual_labels_button.setStyleSheet(basic_button_style)
        self.show_manual_labels_proxy = QGraphicsProxyWidget()
        self.show_manual_labels_proxy.setWidget(self.show_manual_labels_button)

        # add colormap selection button
        def next_color_state(_):
            self.color_state = np.mod(self.color_state + 1, len(self.color_state_names))
            self.color_button.setText(self.color_state_names[self.color_state])
            self.update_label_colors()

        self.color_button = QPushButton(text=self.color_state_names[self.color_state])
        self.color_button.setCheckable(False)
        self.color_button.clicked.connect(next_color_state)
        self.color_button.setStyleSheet(basic_button_style)
        self.color_proxy = QGraphicsProxyWidget()
        self.color_proxy.setWidget(self.color_button)

        # add colormap selection button
        def next_colormap(_):
            self.idx_colormap = np.mod(self.idx_colormap + 1, len(self.colormaps))
            self.colormap_selection.setText(self.colormaps[self.idx_colormap])
            self.update_label_colors()

        self.colormap_selection = QPushButton(text=self.colormaps[self.idx_colormap])
        self.colormap_selection.clicked.connect(next_colormap)
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

        # add feature plots to napari window
        self.dock_window = self.viewer.window.add_dock_widget(self.feature_window, name="ROI Features", area="bottom")

        def switch_image_label(_):
            self.show_mask_image = not self.show_mask_image
            self.update_visibility()

        def update_mask_visibility(_):
            self.mask_visibility = not self.mask_visibility
            self.update_visibility()

        def update_reference_visibility(_):
            self.reference.visible = not self.reference.visible

        self.viewer.bind_key("t", toggle_cells_to_view, overwrite=True)
        self.viewer.bind_key("s", switch_image_label, overwrite=True)
        self.viewer.bind_key("v", update_mask_visibility, overwrite=True)
        self.viewer.bind_key("r", update_reference_visibility, overwrite=True)
        self.viewer.bind_key("c", next_color_state, overwrite=True)
        self.viewer.bind_key("a", next_colormap, overwrite=True)
        self.viewer.bind_key("Control-c", self.save_selection, overwrite=False)

        # create single-click callback for printing data about ROI features
        def single_click_label(_, event):
            if not self.labels.visible:
                self.viewer.status = "can only manually select cells when the labels are visible!"
                return

            # get click data
            plane_idx, yidx, xidx = [int(pos) for pos in event.position]
            label_idx = self.labels.data[plane_idx, yidx, xidx]
            if label_idx == 0:
                self.viewer.status = "single-click on background, no ROI selected"
                return

            # get ROI data
            roi_idx = label_idx - 1  # oh napari, oh napari
            feature_print = [f"{feature}={fvalue[roi_idx]:.3f}" for feature, fvalue in self.rcp.features.items()]

            string_to_print = f"ROI: {roi_idx}" + " ".join(feature_print)

            # only print single click data if alt is held down
            if "Alt" in event.modifiers:
                print(string_to_print)

            # always show message in viewer status
            self.viewer.status = string_to_print

        def double_click_label(_, event):
            self.viewer.status = "you just double clicked!"  # will be overwritten - useful for debugging

            # if not looking at labels, then don't allow manual selection (it would be random!)
            if not self.labels.visible:
                self.viewer.status = "can only manually select cells when the labels are visible!"
                return

            # if not looking at manual annotations, don't allow manual selection...
            if not self.use_manual_labels:
                self.viewer.status = "can only manually select cells when the manual labels are being used!"
                return

            plane_idx, yidx, xidx = [int(pos) for pos in event.position]
            label_idx = self.labels.data[plane_idx, yidx, xidx]
            if label_idx == 0:
                self.viewer.status = "double-click on background, no ROI identity toggled"
            else:
                if "Alt" in event.modifiers:
                    self.viewer.status = "Alt was used, assuming you are trying to single click and not doing a manual label!"
                else:
                    roi_idx = label_idx - 1
                    if "Control" in event.modifiers:
                        if self.only_manual_labels:
                            self.manual_label_active[roi_idx] = False
                            self.viewer.status = f"you just removed the manual label from roi: {roi_idx}"
                        else:
                            self.viewer.status = f"you can only remove a label if you are only looking at manual labels!"
                    else:
                        # manual annotation: if plotting control cells, then annotate as red (1), if plotting red cells, annotate as control (0)
                        new_label = copy(self.show_control_cells)
                        self.manual_label[roi_idx] = new_label
                        self.manual_label_active[roi_idx] = True
                        self.viewer.status = f"you just labeled roi: {roi_idx} with the identity: {new_label}"
                    self.regenerate_mask_data()

        self.labels.mouse_drag_callbacks.append(single_click_label)
        self.masks.mouse_drag_callbacks.append(single_click_label)
        self.reference.mouse_drag_callbacks.append(single_click_label)
        self.labels.mouse_double_click_callbacks.append(double_click_label)
        self.masks.mouse_double_click_callbacks.append(double_click_label)
        self.reference.mouse_double_click_callbacks.append(double_click_label)

        # add callback for dimension slider
        def update_plane_idx(event):
            self.plane_idx = event.source.current_step[0]
            self.update_feature_plots()

        self.viewer.dims.events.connect(update_plane_idx)

    def update_visibility(self):
        """Update the visibility of the masks and labels in the napari viewer."""
        self.masks.visible = self.show_mask_image and self.mask_visibility
        self.labels.visible = not self.show_mask_image and self.mask_visibility

    def update_feature_plots(self):
        """Update the histograms of the intensity features of the red cells in the napari viewer."""
        for feature in self.rcp.features:
            self.hist_graphs[feature].setOpts(height=self.h_values[feature][self.plane_idx])
            self.hist_reds[feature].setOpts(height=self.h_values_red[feature][self.plane_idx])

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
            colors = plt.colormaps[self.colormaps[self.idx_colormap]](norm(self.rcp.features[color_state_name]))
            color_dict = dict(zip(1 + np.arange(self.rcp.num_rois), colors))
            color_dict[None] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.single)  # transparent background (or default)
            colormap = direct_colormap(color_dict)
        # Update colors of the labels
        self.labels.colormap = colormap

    def update_by_feature_criterion(self):
        """Update the idx of cells meeting the criterion defined by the features."""
        self.red_idx = np.full(self.rcp.num_rois, True)  # start with all as red
        for feature, value in self.rcp.features.items():
            if self.feature_active[feature][0]:
                # only keep in red_idx if above minimum
                self.red_idx &= value >= self.feature_cutoffs[feature][0]
            if self.feature_active[feature][1]:
                # only keep in red_idx if below maximum
                self.red_idx &= value <= self.feature_cutoffs[feature][1]

        self.regenerate_mask_data()

    def regenerate_mask_data(self):
        """Regenerate the mask image and labels in the napari viewer based on the current selection.

        Used whenever the selection of red cells is updated or the manual labels are updated such that
        the GUI reflects the current selection appropriately.
        """
        self.masks.data = self.mask_image
        self.labels.data = self.mask_labels
        features_by_plane = {key: utils.split_planes(value, self.rcp.rois_per_plane) for key, value in self.rcp.features.items()}
        idx_selected_by_plane = utils.split_planes(self.idx_selected_masks, self.rcp.rois_per_plane)
        for feature in self.rcp.features:
            for iplane in range(self.rcp.num_planes):
                c_feature_values = features_by_plane[feature][iplane][idx_selected_by_plane[iplane]]
                self.h_values_red[feature][iplane] = np.histogram(c_feature_values, bins=self.h_bin_edges[feature])[0]

        # regenerate histograms
        for feature in self.rcp.features:
            self.hist_reds[feature].setOpts(height=self.h_values_red[feature][self.plane_idx])

    def save_selection(self):
        print("saving red cell curation choices... --- but not implemented!!!!")
        return
        # fullRedIdx = np.concatenate(self.redIdx)
        # fullManualLabels = np.stack((np.concatenate(self.manualLabel), np.concatenate(self.manualLabelActive)))
        # self.redCell.saveone(fullRedIdx, "mpciROIs.redCellIdx")
        # self.redCell.saveone(fullManualLabels, "mpciROIs.redCellManualAssignments")
        # for idx, name in enumerate(self.rcp.features):
        #     cFeatureCutoffs = self.featureCutoffs[idx]
        #     if not (self.feature_active[idx][0]):
        #         cFeatureCutoffs[0] = np.nan
        #     if not (self.feature_active[idx][1]):
        #         cFeatureCutoffs[1] = np.nan
        #     self.redCell.saveone(self.featureCutoffs[idx], self.redCell.oneNameFeatureCutoffs(name))

        # print(f"Red Cell curation choices are saved for session {self.redCell.sessionPrint()}")

    # ------------------------------
    # --------- properties ---------
    # ------------------------------
    @property
    def mask_image(self):
        """Return the masks in each volume as an image by summing the selected masks in each plane.

        Each pixel in the image is the sum of the intensity footprints of the selected
        masks in each plane. Masks are included in the sum if they are selected by the
        user (i.e. True in idx_selected_masks).

        Returns
        -------
        mask_image_by_plane : np.ndarray (float)
            The mask image for each plane in the volume. Filters the masks in each plane
            by idx_selected_masks and sums over their intensity footprints to create a
            single image for each plane.
        """
        idx_selected = self.idx_selected_masks
        image_data = np.zeros((self.rcp.num_planes, self.rcp.ly, self.rcp.lx), dtype=float)
        for iroi, (plane, lam, ypix, xpix) in enumerate(zip(self.rcp.plane_idx, self.rcp.lam, self.rcp.ypix, self.rcp.xpix)):
            if idx_selected[iroi]:
                image_data[plane, ypix, xpix] += lam
        return image_data

    @property
    def mask_labels(self):
        """Return the masks in each volume as labels by summing the selected masks in each plane.

        ROIs are assigned an index that is unique across all ROIs independent of plane.
        The index is offset by 1 because napari uses 0 to indicate "no label". ROIs are
        only presented if the are currently "selected" by the user (i.e. True in
        idx_selected_masks).

        Returns
        -------
        mask_labels_by_plane : np.ndarray (int)
            Each pixel is assigned a label associated with each ROI. The label is the
            index to the ROI - and if ROIs are overlapping then the last ROI will be
            used. Only ROIs that are selected by the user are included in the labels.
        """
        idx_selected = self.idx_selected_masks
        label_data = np.zeros((self.rcp.num_planes, self.rcp.ly, self.rcp.lx), dtype=int)
        for iroi, (plane, ypix, xpix) in enumerate(zip(self.rcp.plane_idx, self.rcp.ypix, self.rcp.xpix)):
            if idx_selected[iroi]:
                label_data[plane, ypix, xpix] = iroi + 1
        return label_data

    @property
    def idx_selected_masks(self):
        """Return a boolean index of the currently selected masks.

        Returns
        -------
        idx_selected_masks_by_plane : np.ndarray (bool)
            The indices of the selected masks across planes.
        """
        if self.only_manual_labels:
            idx = np.full(self.rcp.num_rois, False)
        else:
            if self.show_control_cells:
                idx = np.copy(~self.red_idx)
            else:
                idx = np.copy(self.red_idx)
        if self.use_manual_labels:
            idx[self.manual_label_active] = self.manual_label[self.manual_label_active] != self.show_control_cells
        return idx
