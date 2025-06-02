import pytest
import numpy as np
from PyQt5 import QtWidgets
import os
from cellector.gui import SelectionGUI
from cellector.gui.utils import SelectionState, SelectionConfig
from cellector.roi_processor import RoiProcessor

SKIP_GUI = os.environ.get("SKIP_GUI_TESTS", "1") == "1"


@pytest.fixture(scope="session", autouse=True)
def headless_qt_and_napari_env():
    """Configure headless mode for Qt and Napari for GUI tests."""
    # Save original environment variables
    original_qt_platform = os.environ.get("QT_QPA_PLATFORM")
    original_napari_testing = os.environ.get("NAPARI_TESTING")

    # Set headless environment
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["NAPARI_TESTING"] = "true"

    yield

    # Restore original environment variables
    if original_qt_platform is not None:
        os.environ["QT_QPA_PLATFORM"] = original_qt_platform
    else:
        os.environ.pop("QT_QPA_PLATFORM", None)

    if original_napari_testing is not None:
        os.environ["NAPARI_TESTING"] = original_napari_testing
    else:
        os.environ.pop("NAPARI_TESTING", None)


@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary test data directory."""
    return tmp_path


@pytest.fixture
def sample_mask_volume():
    """Create a sample 3D mask volume for testing."""
    # Create a 3D volume with 3 ROIs, each 10x10 pixels
    volume = np.zeros((3, 10, 10))

    # ROI 1: 3x3 square in top-left
    volume[0, 1:4, 1:4] = np.random.rand(3, 3)

    # ROI 2: 3x3 square in bottom-right
    volume[1, 6:9, 6:9] = np.random.rand(3, 3)

    # ROI 3: 2x4 rectangle in middle
    volume[2, 4:6, 3:7] = np.random.rand(2, 4)

    return volume


@pytest.fixture
def sample_reference():
    """Create a sample reference image for testing."""
    # Create a 3D reference volume with 2 planes, each 10x10 pixels
    return np.random.rand(2, 10, 10)


@pytest.fixture
def sample_functional_reference():
    """Create a sample functional reference image for testing."""
    # Create a 3D functional reference volume with 2 planes, each 10x10 pixels
    return np.random.rand(2, 10, 10)


@pytest.fixture
def sample_plane_idx():
    """Create sample plane indices for testing."""
    # Assign ROIs to planes: first two in plane 0, last in plane 1
    return np.array([0, 0, 1])


@pytest.fixture
def sample_roi_processor(
    test_data_dir,
    sample_mask_volume,
    sample_reference,
    sample_functional_reference,
    sample_plane_idx,
):
    """Create a sample RoiProcessor for testing."""
    return RoiProcessor(
        test_data_dir,
        sample_plane_idx,
        [np.where(mask)[0] for mask in sample_mask_volume],
        [np.where(mask)[1] for mask in sample_mask_volume],
        [mask[np.where(mask)] for mask in sample_mask_volume],
        sample_reference,
        functional_reference=sample_functional_reference,
    )


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app
    app.quit()


@pytest.fixture
def selection_gui(sample_roi_processor, qapp):
    """Create a SelectionGUI instance for testing."""
    gui = None
    try:
        gui = SelectionGUI(sample_roi_processor)
        yield gui
    except Exception:
        raise
    finally:
        if gui is not None and hasattr(gui, "viewer"):
            gui.viewer.close()


@pytest.mark.skipif(SKIP_GUI, reason="Skipping GUI tests in CI/headless environments.")
def test_gui_initialization(selection_gui):
    """Test that the GUI initializes correctly."""
    assert isinstance(selection_gui, SelectionGUI)
    assert isinstance(selection_gui.state, SelectionState)
    assert selection_gui.num_bins == SelectionConfig.DEFAULT_NUM_BINS
    assert selection_gui.roi_processor.num_rois == 3


@pytest.mark.skipif(SKIP_GUI, reason="Skipping GUI tests in CI/headless environments.")
def test_gui_layers(selection_gui):
    """Test that all required layers are created."""
    # Check if viewer is properly initialized
    assert hasattr(selection_gui, "viewer"), "Viewer not initialized"
    assert selection_gui.viewer is not None, "Viewer is None"

    # Check each layer individually with more informative error messages
    assert "reference" in selection_gui.viewer.layers, "Reference layer not found"
    assert (
        "functional_reference" in selection_gui.viewer.layers
    ), "Functional reference layer not found"
    assert "masks_image" in selection_gui.viewer.layers, "Masks image layer not found"
    assert "mask_labels" in selection_gui.viewer.layers, "Mask labels layer not found"


@pytest.mark.skipif(SKIP_GUI, reason="Skipping GUI tests in CI/headless environments.")
def test_state_toggles(selection_gui):
    """Test the state toggle functions."""
    # Test cell view toggle
    initial_state = selection_gui.state.show_control_cells
    selection_gui.state.toggle_cell_view()
    assert selection_gui.state.show_control_cells != initial_state

    # Test mask type toggle
    initial_state = selection_gui.state.show_mask_image
    selection_gui.state.toggle_mask_type()
    assert selection_gui.state.show_mask_image != initial_state

    # Test mask visibility toggle
    initial_state = selection_gui.state.mask_visibility
    selection_gui.state.toggle_mask_visibility()
    assert selection_gui.state.mask_visibility != initial_state

    # Test reference type toggle
    initial_state = selection_gui.state.show_functional_reference
    selection_gui.state.toggle_reference_type()
    assert selection_gui.state.show_functional_reference != initial_state


@pytest.mark.skipif(SKIP_GUI, reason="Skipping GUI tests in CI/headless environments.")
def test_mask_image_property(selection_gui):
    """Test the mask_image property."""
    mask_image = selection_gui.mask_image
    assert isinstance(mask_image, np.ndarray)
    assert mask_image.shape == (2, 10, 10)  # (planes, height, width)


@pytest.mark.skipif(SKIP_GUI, reason="Skipping GUI tests in CI/headless environments.")
def test_mask_labels_property(selection_gui):
    """Test the mask_labels property."""
    mask_labels = selection_gui.mask_labels
    assert isinstance(mask_labels, np.ndarray)
    assert mask_labels.shape == (2, 10, 10)  # (planes, height, width)
    assert np.max(mask_labels) <= 3  # Should have at most 3 unique labels (one per ROI)


@pytest.mark.skipif(SKIP_GUI, reason="Skipping GUI tests in CI/headless environments.")
def test_feature_window_components(selection_gui):
    """Test that all feature window components are created."""
    assert hasattr(selection_gui, "feature_window")
    assert hasattr(selection_gui, "text_area")
    assert hasattr(selection_gui, "feature_layout")
    assert hasattr(selection_gui, "button_area")
    assert hasattr(selection_gui, "buttons")


@pytest.mark.skipif(SKIP_GUI, reason="Skipping GUI tests in CI/headless environments.")
def test_button_creation(selection_gui):
    """Test that all required buttons are created."""
    required_buttons = {
        "save",
        "toggle_cells",
        "toggle_reference",
        "use_manual_labels",
        "show_manual",
        "clear_manual",
        "color",
        "colormap",
        "enable_all",
    }
    assert set(selection_gui.buttons.keys()) == required_buttons


@pytest.mark.skipif(SKIP_GUI, reason="Skipping GUI tests in CI/headless environments.")
def test_histogram_creation(selection_gui):
    """Test that histograms are created for features."""
    # Check that histograms are created for each feature
    num_features = len(selection_gui.roi_processor.features)
    assert len(selection_gui._hidden_features) == 0  # Initially no hidden features
    assert len(selection_gui.hist_graphs) == num_features
    assert len(selection_gui.hist_cursor) == num_features
    assert len(selection_gui.hist_plots) == num_features
