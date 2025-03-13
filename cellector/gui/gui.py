from copy import copy
import functools
import numpy as np
from PyQt5 import QtCore, QtWidgets
import napari
import pyqtgraph as pg
from napari.utils.colormaps import label_colormap, direct_colormap
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..roi_processor import RoiProcessor
from ..manager import CellectorManager
from .. import utils
from .utils import SelectionState, SelectionConfig, GUIComponentFactory, Event, Layer

gui_factory = GUIComponentFactory()


class SelectionGUI:
    """A GUI for selecting cells based on features in reference to a fluorescence image.

    This GUI allows users to interactively select cells based on feature-based filters
    and manual selection. It displays histograms of intensity features and allows range
    selection for qualifying cells as targets or controls.

    Parameters
    ----------
    roi_processor : RoiProcessor
        An instance of the RoiProcessor class containing masks and fluorescence data
    num_bins : int, optional
        Number of bins for feature histograms, default 50
    """

    def __init__(
        self,
        roi_processor: RoiProcessor,
        num_bins: int = SelectionConfig.DEFAULT_NUM_BINS,
    ):
        if not isinstance(roi_processor, RoiProcessor):
            raise ValueError("roi_processor must be an instance of RoiProcessor")

        # Initialize core components
        self.roi_processor = roi_processor
        self.manager = CellectorManager.make_from_roi_processor(roi_processor)
        self.state = SelectionState()

        # Attributes for setting up the GUI
        self.color_state_names = ["random", *self.manager.features.keys()]
        self.num_bins = num_bins

        # Initialize napari viewer and GUI
        self._initialize_napari_viewer()

    def _initialize_napari_viewer(self) -> None:
        """Initialize the napari viewer and associated GUI components."""
        self.viewer = napari.Viewer(title="Cell Curation")

        # Add image layers
        self._init_image_layers()

        # Build feature window components
        self._init_feature_window()

        # Add keyboard and mouse bindings
        self._init_bindings()

    def _init_image_layers(self) -> None:
        """Initialize the image layers in the napari viewer."""
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
            visible=self.state.show_mask_image,
        )

        self.labels = self.viewer.add_labels(
            self.mask_labels,
            name="mask_labels",
            blending="additive",
            visible=not self.state.show_mask_image,
        )

        # Set current step to be consistent with self.state.plane_idx
        self.viewer.dims.current_step = (
            self.state.plane_idx,
            self.viewer.dims.current_step[1],
            self.viewer.dims.current_step[2],
        )

    def _init_feature_window(self) -> None:
        """Initialize the feature window and its components."""
        # Create main layout container
        self.feature_window = pg.GraphicsLayoutWidget()

        # Create main layout sections
        self.text_area = pg.LabelItem(
            "Welcome to the cell selector GUI", justify="left"
        )
        self.toggle_area = pg.GraphicsLayout()
        self.plot_area = pg.GraphicsLayout()
        self.button_area = pg.GraphicsLayout()

        # Add sections to main window
        self.feature_window.addItem(self.text_area, row=0, col=0)
        self.feature_window.addItem(self.toggle_area, row=1, col=0)
        self.feature_window.addItem(self.plot_area, row=2, col=0)
        self.feature_window.addItem(self.button_area, row=3, col=0)

        # Build UI components
        self._compute_histograms()
        self._build_histograms()
        self._build_cutoff_lines()
        self._build_feature_toggles()
        self._build_buttons()

        # Add feature window to napari viewer
        self.dock_window = self.viewer.window.add_dock_widget(
            self.feature_window, name="ROI Features", area="bottom"
        )

    def _build_buttons(self) -> None:
        """Build the control buttons using GUIComponentFactory."""
        buttons = {
            "save": ("Save Selection", self.save_selection),
            "toggle_cells": ("Target Cells", self._toggle_cells_to_view),
            "use_manual_labels": (
                "Using Manual Labels",
                self._toggle_use_manual_labels,
            ),
            "show_manual": ("All Labels", self._show_manual_labels),
            "clear_manual": ("Clear Manual Labels", self._clear_manual_labels),
            "color": (
                self.color_state_names[self.state.color_state],
                self._next_color_state,
            ),
            "colormap": (
                SelectionConfig.COLORMAPS[self.state.idx_colormap],
                self._next_colormap,
            ),
        }

        self.buttons = {}
        self.button_proxies = {}
        for idx, (name, (text, callback)) in enumerate(buttons.items()):
            button, proxy = gui_factory.create_button(
                text=text, callback=callback, style=SelectionConfig.STYLES["BUTTON"]
            )
            self.buttons[name] = button
            self.button_proxies[name] = proxy
            self.button_area.addItem(proxy, row=0, col=idx)

    @property
    def mask_image(self) -> np.ndarray:
        """Generate mask image from ROIs in the cursor.

        Each pixel in the image is the sum of the intensity footprints of the masks in each
        the cursor in each plane. Masks are included in the sum if they are True in
        idx_cursor.

        Returns
        -------
        mask_image_by_plane : np.ndarray (float)
            The mask image for each plane in the volume. Filters the masks in each plane
            by idx_cursor and sums over their intensity footprints to create a
            single image for each plane.
        """
        idx_cursor = self.idx_cursor
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
            if idx_cursor[iroi]:
                image_data[plane, ypix, xpix] += lam

        return image_data

    @property
    def mask_labels(self) -> np.ndarray:
        """Generate label image from ROIs in the cursor.

        ROIs are assigned an index that is unique across all ROIs independent of plane.
        The index is offset by 1 because napari uses 0 to indicate "no label". ROIs are
        only presented if the are True in idx_cursor.

        Returns
        -------
        mask_labels_by_plane : np.ndarray (int)
            Each pixel is assigned a label associated with each ROI. The label is the
            index to the ROI - and if ROIs are overlapping then the last ROI will be
            used. Only ROIs that are True in idx_cursor are included.
        """
        idx_cursor = self.idx_cursor
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
            if idx_cursor[iroi]:
                label_data[plane, ypix, xpix] = iroi + 1

        return label_data

    @property
    def idx_cursor(self):
        """Return a boolean index of the masks in the cursor.

        The terminology here is that the cursor is a subset of the masks that are
        currently being shown based on the feature cutoffs, manual labels, and the state
        of the GUI (e.g. if the user wants to show control cells, target cells, only
        manual cells, etc).

        Returns
        -------
        idx_cursor_by_plane : np.ndarray (bool)
            The indices of the cursor masks across planes.
        """
        if self.state.only_manual_labels:
            # if only manual labels, ignore the feature criteria
            idx = np.full(self.roi_processor.num_rois, False)
        else:
            # otherwise, use the feature criteria
            if self.state.show_control_cells:
                idx = np.copy(~self.manager.idx_meets_criteria)
            else:
                idx = np.copy(self.manager.idx_meets_criteria)
        if self.state.use_manual_labels:
            # Of the ROIs with active labels
            # If manual_label is set to True, then the ROI is a target cell, not a control cell
            # So update the cursor index to manual_label != show_control_cells so that we either show manual target or manual control cells
            idx[self.manager.manual_label_active] = (
                self.manager.manual_label[self.manager.manual_label_active]
                != self.state.show_control_cells
            )
        return idx

    def _init_bindings(self) -> None:
        """Initialize keyboard and mouse bindings for the viewer."""
        # Keyboard bindings
        key_bindings = {
            "t": self._toggle_cells_to_view,
            "s": self._switch_image_label,
            "v": self._update_mask_visibility,
            "r": self._update_reference_visibility,
            "c": self._next_color_state,
            "a": self._next_colormap,
        }

        for key, callback in key_bindings.items():
            self.viewer.bind_key(key, callback, overwrite=True)

        # Mouse bindings
        for layer in [self.labels, self.masks, self.reference]:
            layer.mouse_drag_callbacks.append(self._single_click_label)
            layer.mouse_double_click_callbacks.append(self._double_click_label)

        # Dimension change binding
        self.viewer.dims.events.connect(self._update_plane_idx)

    def _compute_histograms(self) -> None:
        """Compute histograms for all features."""
        self.h_values_full = {}
        self.h_values_cursor = {}
        self.h_bin_edges = {}
        self.h_values_maximum = {}

        idx_cursor_by_plane = utils.split_planes(
            self.idx_cursor, self.roi_processor.rois_per_plane
        )
        for feature_name, feature_values in self.manager.features.items():
            # Compute bin edges for the feature
            edges = np.histogram(feature_values, bins=self.num_bins)[1]
            self.h_bin_edges[feature_name] = edges

            # Compute histograms for each plane
            features_by_plane = utils.split_planes(
                feature_values, self.roi_processor.rois_per_plane
            )
            cursor_by_plane = [
                fbp[idx] for fbp, idx in zip(features_by_plane, idx_cursor_by_plane)
            ]
            self.h_values_full[feature_name] = [
                np.histogram(plane_values, bins=edges)[0]
                for plane_values in features_by_plane
            ]
            self.h_values_cursor[feature_name] = [
                np.histogram(plane_values, bins=edges)[0]
                for plane_values in cursor_by_plane
            ]
            self.h_values_maximum[feature_name] = np.max(
                np.concatenate(self.h_values_full[feature_name])
            )  # Maximum is max counts across all planes

    def _build_histograms(self) -> None:
        """Build histograms for feature visualization."""
        self.hist_layout = pg.GraphicsLayout()
        self.hist_graphs = {}
        self.hist_cursor = {}

        for feature in self.manager.features:
            # Create histogram bars
            full_hist = gui_factory.create_histogram(
                data=self.h_values_full[feature][self.state.plane_idx],
                bins=self.h_bin_edges[feature],
            )

            cursor_hist = gui_factory.create_histogram(
                data=self.h_values_cursor[feature][self.state.plane_idx],
                bins=self.h_bin_edges[feature],
                color="r",
            )

            self.hist_graphs[feature] = full_hist
            self.hist_cursor[feature] = cursor_hist

        # Then build the plot widgets
        self.hist_plots = {}
        self.preserve_methods = {}

        for idx, feature in enumerate(self.manager.features):
            # Create plot
            plot = self.plot_area.addPlot(row=0, col=idx, title=feature)
            plot.setMouseEnabled(x=False)
            plot.setYRange(0, self.h_values_maximum[feature])

            # Add histogram items
            plot.addItem(self.hist_graphs[feature])
            plot.addItem(self.hist_cursor[feature])

            # Set up y-range preservation
            self.preserve_methods[feature] = functools.partial(
                self._preserve_y_range, feature=feature
            )
            plot.getViewBox().sigYRangeChanged.connect(self.preserve_methods[feature])

            # And add it to the dictionary of histogram plot widgets
            self.hist_plots[feature] = plot

    def _build_cutoff_lines(self) -> None:
        """Build cutoff lines for feature selection."""
        self.feature_range = {}
        self.feature_active = {}
        self.feature_cutoffs = {}
        self.cutoff_lines = {}

        for feature in self.manager.features:
            criterion = self.manager.criteria[feature]
            self.feature_range[feature] = [
                np.min(self.h_bin_edges[feature]),
                np.max(self.h_bin_edges[feature]),
            ]
            self.feature_active[feature] = [
                criterion[0] is not None,
                criterion[1] is not None,
            ]
            self.feature_cutoffs[feature] = copy(criterion)
            # But if no criteria, set to the range
            if self.feature_cutoffs[feature][0] is None:
                self.feature_cutoffs[feature][0] = self.feature_range[feature][0]
            if self.feature_cutoffs[feature][1] is None:
                self.feature_cutoffs[feature][1] = self.feature_range[feature][1]
            self.feature_cutoffs[feature] = sorted(self.feature_cutoffs[feature])

            cutoff_lines = []
            for i in range(2):
                line = pg.InfiniteLine(
                    pos=self.feature_cutoffs[feature][i], movable=True
                )
                line.setBounds(self.feature_range[feature])
                line.sigPositionChangeFinished.connect(
                    functools.partial(self._update_cutoff_finished, feature=feature)
                )
                cutoff_lines.append(line)
                self.hist_plots[feature].addItem(line)
            self.cutoff_lines[feature] = cutoff_lines

    def _build_feature_toggles(self) -> None:
        """Build toggle buttons for feature selection."""
        self.min_max_name = ["min", "max"]
        self.max_length_name = (
            max(len(feature) for feature in self.manager.features) + 9
        )

        self.use_feature_buttons = {}
        self.use_feature_proxies = []

        for idx, feature in enumerate(self.manager.features):
            feature_toggles = []
            for i in range(2):
                proxy_idx = 2 * idx + i
                is_active = self.feature_active[feature][i]

                button, proxy = gui_factory.create_button(
                    text=self._get_cutoff_toggle_text(feature, i, is_active),
                    callback=functools.partial(
                        self._toggle_feature, feature=feature, iminmax=i
                    ),
                    style=SelectionConfig.STYLES[
                        "UNCHECKED" if is_active else "CHECKED"
                    ],
                    checkable=True,
                )
                button.setChecked(is_active)

                feature_toggles.append(button)
                self.use_feature_proxies.append(proxy)
                self.toggle_area.addItem(proxy, row=0, col=proxy_idx)

            self.use_feature_buttons[feature] = feature_toggles

    def _get_cutoff_toggle_text(self, feature: str, idx: int, is_active: bool) -> str:
        """Generate text for feature cutoff toggle buttons."""
        action = "using" if is_active else "ignore"
        return f"{action} {self.min_max_name[idx]} {feature}".center(
            self.max_length_name, " "
        )

    def regenerate_mask_data(self) -> None:
        """Update mask visualization and histograms based on current selection."""
        # Update mask and label data
        self.masks.data = self.mask_image
        self.labels.data = self.mask_labels

        # Update histograms for cells in the cursor
        features_by_plane = {
            key: utils.split_planes(value, self.roi_processor.rois_per_plane)
            for key, value in self.manager.features.items()
        }
        idx_cursor_by_plane = utils.split_planes(
            self.idx_cursor, self.roi_processor.rois_per_plane
        )

        for feature in self.manager.features:
            for iplane in range(self.roi_processor.num_planes):
                c_feature_values = features_by_plane[feature][iplane][
                    idx_cursor_by_plane[iplane]
                ]
                self.h_values_cursor[feature][iplane] = np.histogram(
                    c_feature_values, bins=self.h_bin_edges[feature]
                )[0]

        # Update histogram visualizations
        self.update_feature_plots()

    def update_feature_plots(self) -> None:
        """Update the histogram plots for all features."""
        for feature in self.manager.features:
            self.hist_graphs[feature].setOpts(
                height=self.h_values_full[feature][self.state.plane_idx]
            )
            self.hist_cursor[feature].setOpts(
                height=self.h_values_cursor[feature][self.state.plane_idx]
            )

    def update_label_colors(self) -> None:
        """Update the colors of the label visualization."""
        color_state_name = self.color_state_names[self.state.color_state]

        if color_state_name == "random":
            # this is inherited from the default random colormap in napari
            colormap = label_colormap(49, 0.5, background_value=0)
        else:
            # Create colormap based on feature values
            norm = mpl.colors.Normalize(
                vmin=self.feature_range[color_state_name][0],
                vmax=self.feature_range[color_state_name][1],
            )
            colors = plt.colormaps[SelectionConfig.COLORMAPS[self.state.idx_colormap]](
                norm(self.manager.features[color_state_name])
            )

            # Create color dictionary with transparent background
            color_dict = dict(zip(1 + np.arange(self.manager.num_rois), colors))
            color_dict[None] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.single)
            colormap = direct_colormap(color_dict)

        self.labels.colormap = colormap

    def update_visibility(self) -> None:
        """Update the visibility of masks and labels."""
        self.masks.visible = self.state.show_mask_image and self.state.mask_visibility
        self.labels.visible = (
            not self.state.show_mask_image and self.state.mask_visibility
        )

    def save_selection(self) -> None:
        """Save the current selection state."""
        self.manager.save_all()
        self.update_text("Selection saved!")

    def update_text(self, text: str) -> None:
        """Update the text display in both text area and status bar."""
        self.text_area.setText(text)
        self.viewer.status = text

    def _preserve_y_range(self, feature: str) -> None:
        """Preserve useful y-axis range for histogram plots.

        Parameters
        ----------
        feature : str
            The feature whose plot range is being preserved
        """
        # Temporarily disconnect to prevent recursive calls
        self.hist_plots[feature].getViewBox().sigYRangeChanged.disconnect(
            self.preserve_methods[feature]
        )

        # Calculate new range
        current_min, current_max = self.hist_plots[feature].viewRange()[1]
        current_range = current_max - current_min
        new_max = min(current_range, self.h_values_maximum[feature])

        # Update range and reconnect signal
        self.hist_plots[feature].setYRange(0, new_max)
        self.hist_plots[feature].getViewBox().sigYRangeChanged.connect(
            self.preserve_methods[feature]
        )

    def _update_cutoff_finished(self, event: Event, feature: str) -> None:
        """Handle completion of cutoff line movement.

        Parameters
        ----------
        event : Event
            The event trigger (unused)
        feature : str
            The feature whose cutoff was updated
        """
        # Get and sort cutoff values
        cutoff_values = sorted(
            [
                self.cutoff_lines[feature][0].pos()[0],
                self.cutoff_lines[feature][1].pos()[0],
            ]
        )
        self.feature_cutoffs[feature] = cutoff_values

        # Update UI elements
        self.cutoff_lines[feature][0].setValue(cutoff_values[0])
        self.cutoff_lines[feature][1].setValue(cutoff_values[1])

        # Update feature activity state
        for i in range(2):
            is_at_limit = cutoff_values[i] == self.feature_range[feature][i]
            self.feature_active[feature][i] = not is_at_limit
            self._update_feature_toggle(feature, i)

        self._update_criteria(feature)

    def _toggle_feature(self, event: Event, feature: str, iminmax: int) -> None:
        """Handle feature toggle button clicks.

        Parameters
        ----------
        event : Event
            The event trigger (unused)
        feature : str
            The feature being toggled
        iminmax : int
            Index indicating minimum (0) or maximum (1) toggle
        """
        is_active = self.use_feature_buttons[feature][iminmax].isChecked()
        self.feature_active[feature][iminmax] = is_active
        self._update_feature_toggle(feature, iminmax)
        self._update_criteria(feature)

    def _update_feature_toggle(self, feature: str, iminmax: int) -> None:
        """Update feature toggle button state and appearance.

        Parameters
        ----------
        feature : str
            The feature to update
        iminmax : int
            Index indicating minimum (0) or maximum (1) toggle
        """
        is_active = self.feature_active[feature][iminmax]
        button = self.use_feature_buttons[feature][iminmax]

        if is_active:
            button.setText(self._get_cutoff_toggle_text(feature, iminmax, True))
            button.setStyleSheet(SelectionConfig.STYLES["UNCHECKED"])
            self.cutoff_lines[feature][iminmax].setValue(
                self.feature_cutoffs[feature][iminmax]
            )
        else:
            button.setText(self._get_cutoff_toggle_text(feature, iminmax, False))
            button.setStyleSheet(SelectionConfig.STYLES["CHECKED"])
            self.cutoff_lines[feature][iminmax].setValue(
                self.feature_range[feature][iminmax]
            )

    def _update_criteria(self, feature: str) -> None:
        """Update selection criteria for a feature.

        Parameters
        ----------
        feature : str
            The feature whose criteria are being updated
        """
        # Get current cutoff values and active states
        cutoff_values = [line.pos()[0] for line in self.cutoff_lines[feature]]
        cutoff_active = self.feature_active[feature]

        # Create criteria
        min_cutoff, max_cutoff = sorted(cutoff_values)
        criteria = [
            min_cutoff if cutoff_active[0] else None,
            max_cutoff if cutoff_active[1] else None,
        ]

        # Update manager and visualization
        self.manager.update_criteria(feature, criteria)
        self.regenerate_mask_data()

    def _toggle_cells_to_view(self, event: Event) -> None:
        """Toggle between showing control and target cells."""
        self.state.toggle_cell_view()
        self.buttons["toggle_cells"].setText(
            "Control Cells" if self.state.show_control_cells else "Target Cells"
        )
        self.regenerate_mask_data()
        self.update_text(
            f"Now viewing {'control' if self.state.show_control_cells else 'target'} cells"
        )

    def _single_click_label(self, layer: Layer, event: Event) -> None:
        """Handle single-click events on labels.

        Parameters
        ----------
        layer : Layer
            The layer that was clicked
        event : Event
            The click event
        """
        if not self.labels.visible:
            self.update_text("Can only click on cells when labels are visible!")
            return

        # Get click location and label
        plane_idx, yidx, xidx = [int(pos) for pos in event.position]
        label_idx = self.labels.data[plane_idx, yidx, xidx]

        if label_idx == 0:
            self.update_text("Single-click on background, no ROI selected")
            return

        # Get ROI data and format feature information
        roi_idx = label_idx - 1
        feature_info = [
            f"{feature}={fvalue[roi_idx]:.3f}"
            for feature, fvalue in self.manager.features.items()
        ]

        status_message = f"ROI: {roi_idx} " + " ".join(feature_info)

        # Print detailed info if Alt is held
        if "Alt" in event.modifiers:
            print(status_message)

        self.update_text(status_message)

    def _double_click_label(self, layer: Layer, event: Event) -> None:
        """Handle double-click events for manual cell selection.

        Parameters
        ----------
        layer : Layer
            The layer that was clicked
        event : Event
            The click event
        """
        # Validation checks
        if not self.labels.visible:
            self.update_text("Can only manually select cells when labels are visible!")
            return

        if not self.state.use_manual_labels:
            self.update_text(
                "Can only manually select cells when manual labels are being used!"
            )
            return

        # Get click location and label
        plane_idx, yidx, xidx = [int(pos) for pos in event.position]
        label_idx = self.labels.data[plane_idx, yidx, xidx]

        if label_idx == 0:
            self.update_text("Double-click on background, no ROI selected")
            return

        if "Alt" in event.modifiers:
            self._single_click_label(layer, event)
            return

        roi_idx = label_idx - 1
        self._handle_manual_label_update(roi_idx, event)

    def _handle_manual_label_update(self, roi_idx: int, event: Event) -> None:
        """Update manual labels based on user interaction.

        Parameters
        ----------
        roi_idx : int
            Index of the ROI to update
        event : Event
            The event containing modifier keys
        """
        if "Control" in event.modifiers:
            if self.state.only_manual_labels:
                self.manager.manual_label_active[roi_idx] = False
                self.update_text(
                    f"You just removed the manual label from ROI: {roi_idx}"
                )
            else:
                self.update_text(
                    "You can only remove a label if you are only looking at manual labels!"
                )
        else:
            # Label as target if showing controls, and vice versa
            new_label = copy(bool(self.state.show_control_cells))
            self.manager.manual_label[roi_idx] = new_label
            self.manager.manual_label_active[roi_idx] = True
            self.update_text(
                f"You just labeled ROI: {roi_idx} with the identity: {new_label}"
            )

        self.regenerate_mask_data()

    def _toggle_use_manual_labels(self, event: Event) -> None:
        """Toggle the use of manual labels."""
        self.state.use_manual_labels = not self.state.use_manual_labels

        self.buttons["use_manual_labels"].setText(
            "Using Manual Labels"
            if self.state.use_manual_labels
            else "Ignoring Manual Labels"
        )

        self.regenerate_mask_data()
        self.update_text(
            f"{'Using' if self.state.use_manual_labels else 'Ignoring'} manual labels"
        )

    def _show_manual_labels(self, event: Event) -> None:
        """Toggle between showing all labels or only manual labels."""
        self.state.only_manual_labels = not self.state.only_manual_labels

        if self.state.only_manual_labels:
            self.state.use_manual_labels = True

        self.buttons["show_manual"].setText(
            "Only Manual Labels" if self.state.only_manual_labels else "All Labels"
        )

        self.regenerate_mask_data()
        self.update_text(
            "Only showing manual labels"
            if self.state.only_manual_labels
            else "Showing all labels"
        )

    def _clear_manual_labels(self, event: Event) -> None:
        """Clear all manual labels with confirmation check."""
        modifiers = QtWidgets.QApplication.keyboardModifiers()

        if modifiers == QtCore.Qt.ControlModifier:
            self.manager.manual_label_active[:] = False
            self.regenerate_mask_data()
            self.update_text("You just cleared all manual labels!")
        else:
            self.update_text(
                "Clearing manual labels requires a control click for safety! Try again."
            )

    def _next_color_state(self, event: Event) -> None:
        """Cycle through color states for visualization."""
        self.state.color_state = (self.state.color_state + 1) % len(
            self.color_state_names
        )
        self.buttons["color"].setText(self.color_state_names[self.state.color_state])
        self.update_label_colors()
        self.update_text(
            f"Now coloring by {self.color_state_names[self.state.color_state]}"
        )

    def _next_colormap(self, event: Event) -> None:
        """Cycle through available colormaps."""
        self.state.idx_colormap = (self.state.idx_colormap + 1) % len(
            SelectionConfig.COLORMAPS
        )
        self.buttons["colormap"].setText(
            SelectionConfig.COLORMAPS[self.state.idx_colormap]
        )
        self.update_label_colors()

    def _switch_image_label(self, event: Event) -> None:
        """Toggle between mask image and label visualization."""
        self.state.toggle_mask_type()
        self.update_visibility()
        self.update_text(
            f"Now showing {'mask image' if self.state.show_mask_image else 'mask labels'}"
        )

    def _update_mask_visibility(self, event: Event) -> None:
        """Toggle overall mask visibility."""
        self.state.toggle_mask_visibility()
        self.update_visibility()

    def _update_reference_visibility(self, event: Event) -> None:
        """Toggle visibility of the reference image."""
        self.reference.visible = not self.reference.visible

    def _update_plane_idx(self, event: Event) -> None:
        """Update plane index and coordinate feature histograms with viewer."""
        self.state.plane_idx = event.source.current_step[0]
        self.update_feature_plots()
