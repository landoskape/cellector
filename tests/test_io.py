import pytest
import numpy as np
from pathlib import Path
from cellector.io import (
    create_from_pixel_data,
    create_from_mask_volume,
    create_from_suite2p,
    create_from_suite3d,
)


@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary test data directory."""
    return tmp_path


@pytest.fixture
def mock_suite2p_dir(test_data_dir):
    """Create a mock suite2p directory structure with test data."""
    # Create plane directories
    plane0_dir = test_data_dir / "plane0"
    plane1_dir = test_data_dir / "plane1"
    plane0_dir.mkdir()
    plane1_dir.mkdir()

    # Create mock stats for each plane
    stats0 = [
        {
            "ypix": np.array([1, 2, 3]),
            "xpix": np.array([1, 2, 3]),
            "lam": np.array([0.5, 0.6, 0.7]),
        },
        {
            "ypix": np.array([4, 5, 6]),
            "xpix": np.array([4, 5, 6]),
            "lam": np.array([0.8, 0.9, 1.0]),
        },
    ]
    stats1 = [
        {
            "ypix": np.array([7, 8, 9]),
            "xpix": np.array([7, 8, 9]),
            "lam": np.array([0.3, 0.4, 0.5]),
        }
    ]

    # Create mock ops data with reference images
    ops0 = {
        "meanImg_chan2": np.random.rand(10, 10),  # Reference image
        "meanImg": np.random.rand(10, 10),  # Functional reference
    }
    ops1 = {"meanImg_chan2": np.random.rand(10, 10), "meanImg": np.random.rand(10, 10)}

    # Create mock redcell data
    redcell0 = np.array([[0, 0.8], [1, 0.9]])  # [is_red, probability]
    redcell1 = np.array([[0, 0.7]])

    # Save all data
    np.save(plane0_dir / "stat.npy", stats0)
    np.save(plane1_dir / "stat.npy", stats1)
    np.save(plane0_dir / "ops.npy", ops0)
    np.save(plane1_dir / "ops.npy", ops1)
    np.save(plane0_dir / "redcell.npy", redcell0)
    np.save(plane1_dir / "redcell.npy", redcell1)

    return test_data_dir


@pytest.fixture
def mock_suite3d_dir(test_data_dir):
    """Create a mock suite3d directory structure with test data."""
    # Create mock stats
    stats = [
        {
            "coords": (
                np.array([0, 0, 1, 1]),
                np.array([1, 2, 1, 2]),
                np.array([4, 5, 4, 5]),
            ),
            "lam": np.array([0.5, 0.6, 0.5, 0.6]),
        },
        {
            "coords": (
                np.array([0, 0, 0, 1, 1]),
                np.array([7, 8, 8, 7, 7]),
                np.array([1, 2, 1, 2, 3]),
            ),
            "lam": np.array([0.8, 0.9, 0.8, 0.9, 0.8]),
        },
    ]

    # Create mock reference images
    ref_img_3d_structural = np.random.rand(2, 10, 10)  # [planes, height, width]
    ref_img_3d = np.random.rand(2, 10, 10)

    # Save all data
    np.save(test_data_dir / "stats.npy", stats)
    np.save(test_data_dir / "ref_img_3d_structural.npy", ref_img_3d_structural)
    np.save(test_data_dir / "ref_img_3d.npy", ref_img_3d)

    return test_data_dir


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
def sample_pixel_data(sample_mask_volume):
    """Create sample pixel data from the mask volume."""
    stats = []
    for mask in sample_mask_volume:
        ypix, xpix = np.where(mask)
        lam = mask[ypix, xpix]
        stats.append({"lam": lam, "ypix": ypix, "xpix": xpix})
    return stats


def test_create_from_mask_volume(
    test_data_dir,
    sample_mask_volume,
    sample_reference,
    sample_functional_reference,
    sample_plane_idx,
):
    """Test creating a RoiProcessor from a mask volume."""
    roi_processor = create_from_mask_volume(
        test_data_dir,
        sample_mask_volume,
        sample_reference,
        sample_plane_idx,
        functional_reference=sample_functional_reference,
    )

    # Verify the RoiProcessor was created with correct data
    assert roi_processor.num_rois == len(sample_mask_volume)
    assert roi_processor.reference.shape == sample_reference.shape
    assert roi_processor.functional_reference.shape == sample_functional_reference.shape
    assert roi_processor.volumetric is False


def test_create_from_pixel_data(
    test_data_dir,
    sample_pixel_data,
    sample_reference,
    sample_functional_reference,
    sample_plane_idx,
):
    """Test creating a RoiProcessor from pixel data."""
    roi_processor = create_from_pixel_data(
        test_data_dir,
        sample_pixel_data,
        sample_reference,
        sample_plane_idx,
        functional_reference=sample_functional_reference,
    )

    # Verify the RoiProcessor was created with correct data
    assert roi_processor.num_rois == len(sample_pixel_data)
    assert roi_processor.reference.shape == sample_reference.shape
    assert roi_processor.functional_reference.shape == sample_functional_reference.shape
    assert roi_processor.volumetric is False


def test_create_from_suite2p(mock_suite2p_dir):
    """Test creating a RoiProcessor from a suite2p directory."""
    roi_processor = create_from_suite2p(
        mock_suite2p_dir,
        use_redcell=True,
        reference_key="meanImg_chan2",
        functional_key="meanImg",
    )

    # Verify the RoiProcessor was created with correct data
    assert roi_processor.num_rois == 3  # 2 ROIs in plane0 + 1 ROI in plane1
    assert "red_s2p" in roi_processor.features
    assert roi_processor.reference.shape == (2, 10, 10)  # 2 planes, 10x10 images
    assert roi_processor.functional_reference.shape == (2, 10, 10)
    assert roi_processor.volumetric is False


def test_create_from_suite3d(mock_suite3d_dir):
    """Test creating a RoiProcessor from a suite3d directory."""
    roi_processor = create_from_suite3d(
        mock_suite3d_dir,
        reference_key="ref_img_3d_structural",
        functional_key="ref_img_3d",
    )

    # Verify the RoiProcessor was created with correct data
    assert roi_processor.num_rois == 2
    assert roi_processor.reference.shape == (2, 10, 10)
    assert roi_processor.functional_reference.shape == (2, 10, 10)
    assert roi_processor.volumetric is True
