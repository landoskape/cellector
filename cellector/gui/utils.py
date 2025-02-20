from dataclasses import dataclass
from typing import Tuple, Optional, Protocol, TypedDict
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QGraphicsProxyWidget, QPushButton


@dataclass
class SelectionState:
    """Manages the current state of cell selection and visualization."""

    # Display state
    show_control_cells: bool = False
    show_mask_image: bool = False
    mask_visibility: bool = True

    # Selection state
    use_manual_labels: bool = True
    only_manual_labels: bool = False

    # Visualization state
    color_state: int = 0
    plane_idx: int = 0
    idx_colormap: int = 0

    def toggle_cell_view(self) -> bool:
        """Toggles between showing control and target cells."""
        self.show_control_cells = not self.show_control_cells
        return self.show_control_cells

    def toggle_mask_type(self) -> bool:
        """Toggles between showing mask image and labels."""
        self.show_mask_image = not self.show_mask_image
        return self.show_mask_image

    def toggle_mask_visibility(self) -> bool:
        """Toggles overall mask visibility."""
        self.mask_visibility = not self.mask_visibility
        return self.mask_visibility


class SelectionConfig:
    """Configuration constants for the selection GUI."""

    STYLES = dict(
        BUTTON="""
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
        """,
        CHECKED="""
            QWidget {
                background-color: #1F1F1F;
                color: red;
                font-family: Arial, sans-serif;
            }
        """,
        UNCHECKED="""
            QWidget {
                background-color: #1F1F1F;
                color: #F0F0F0;
                font-family: Arial, sans-serif;
            }
        """,
    )

    COLORMAPS = ["plasma", "autumn", "spring", "summer", "winter", "hot"]
    DEFAULT_NUM_BINS = 50


class GUIComponentFactory:
    """Creates and configures GUI components with consistent styling."""

    @staticmethod
    def create_button(
        text: str,
        callback: callable,
        style: str = SelectionConfig.STYLES["BUTTON"],
        checkable: bool = False,
    ) -> Tuple[QPushButton, QGraphicsProxyWidget]:
        """Creates a styled button with its proxy widget."""
        button = QPushButton(text=text)
        button.clicked.connect(callback)
        button.setStyleSheet(style)
        button.setCheckable(checkable)

        proxy = QGraphicsProxyWidget()
        proxy.setWidget(button)

        return button, proxy

    @staticmethod
    def create_histogram(
        data: np.ndarray, bins: np.ndarray, color: Optional[str] = None
    ) -> pg.BarGraphItem:
        """Creates a histogram bar graph."""
        bar_width = np.diff(bins[:2])
        bin_centers = bins[:-1] + bar_width / 2
        opts = dict(x=bin_centers, height=data, width=bar_width)
        if color:
            opts["brush"] = color
        return pg.BarGraphItem(**opts)


class Layer(Protocol):
    visible: bool
    data: np.ndarray


class Event(TypedDict):
    position: tuple[int, int, int]
    modifiers: set[str]
