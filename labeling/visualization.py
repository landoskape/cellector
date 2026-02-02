"""Visualization utilities for displaying ROI images."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from typing import Optional, Tuple


def get_roi_images(
    roi_processor,
    roi_id: int,
    plane_idx: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract images for a single ROI.

    Parameters
    ----------
    roi_processor : RoiProcessor
        The ROI processor instance.
    roi_id : int
        Index of the ROI to extract.
    plane_idx : Optional[int]
        If provided and volumetric=False, extract from this plane.
        If None and volumetric=False, use the ROI's plane.
        If volumetric=True, this is ignored and all planes are used.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of (fluorescence_image, roi_footprint, combined_image)
        - fluorescence_image: centered reference image
        - roi_footprint: centered mask image
        - combined_image: overlay of both
    """
    # Get centered reference (fluorescence image)
    centered_ref = roi_processor.centered_reference[roi_id]

    # Get centered mask (ROI footprint)
    centered_mask = roi_processor.centered_masks[roi_id]

    # Handle volumetric vs non-volumetric
    if roi_processor.volumetric:
        # Sum across planes for stacked display
        if centered_ref.ndim == 3:
            fluorescence_image = np.sum(centered_ref, axis=0)
        else:
            fluorescence_image = centered_ref

        if centered_mask.ndim == 3:
            roi_footprint = np.sum(centered_mask, axis=0)
        else:
            roi_footprint = centered_mask
    else:
        # Non-volumetric: use the plane specified or ROI's plane
        if plane_idx is None:
            # Get plane from ROI's zpix
            if isinstance(roi_processor.zpix, np.ndarray):
                plane_idx = roi_processor.zpix[roi_id]
            else:
                plane_idx = 0

        if centered_ref.ndim == 3:
            fluorescence_image = centered_ref[plane_idx]
        else:
            fluorescence_image = centered_ref

        if centered_mask.ndim == 3:
            roi_footprint = centered_mask[plane_idx]
        else:
            roi_footprint = centered_mask

    # Create combined image (overlay)
    # Normalize both images for display
    ref_norm = (fluorescence_image - fluorescence_image.min()) / (
        fluorescence_image.max() - fluorescence_image.min() + 1e-10
    )
    mask_norm = (roi_footprint - roi_footprint.min()) / (
        roi_footprint.max() - roi_footprint.min() + 1e-10
    )

    # Create RGB combined image
    # Reference in green channel, mask in red channel
    combined_image = np.zeros((*ref_norm.shape, 3))
    combined_image[:, :, 1] = ref_norm  # Green for fluorescence
    combined_image[:, :, 0] = mask_norm  # Red for ROI footprint
    combined_image[:, :, 2] = (ref_norm + mask_norm) / 2  # Blue as mix
    # Ensure values are in [0, 1] range
    combined_image = np.clip(combined_image, 0, 1)

    return fluorescence_image, roi_footprint, combined_image


def plot_roi_triplet(
    fluorescence_image: np.ndarray,
    roi_footprint: np.ndarray,
    combined_image: np.ndarray,
    colormap: str = "gray",
    vmin_fluo: Optional[float] = None,
    vmax_fluo: Optional[float] = None,
    vmin_mask: Optional[float] = None,
    vmax_mask: Optional[float] = None,
    figsize: Tuple[int, int] = (15, 5),
) -> plt.Figure:
    """Create a (1, 3) subplot showing fluorescence, ROI footprint, and combined.

    Parameters
    ----------
    fluorescence_image : np.ndarray
        The fluorescence/reference image.
    roi_footprint : np.ndarray
        The ROI footprint/mask image.
    combined_image : np.ndarray
        The combined overlay image (RGB).
    colormap : str
        Matplotlib colormap name for grayscale images.
    vmin_fluo : Optional[float]
        Minimum value for fluorescence image scaling.
    vmax_fluo : Optional[float]
        Maximum value for fluorescence image scaling.
    vmin_mask : Optional[float]
        Minimum value for mask image scaling.
    vmax_mask : Optional[float]
        Maximum value for mask image scaling.
    figsize : Tuple[int, int]
        Figure size (width, height).

    Returns
    -------
    plt.Figure
        Matplotlib figure with the three subplots.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Set vmin/vmax defaults
    if vmin_fluo is None:
        vmin_fluo = fluorescence_image.min()
    if vmax_fluo is None:
        vmax_fluo = fluorescence_image.max()
    if vmin_mask is None:
        vmin_mask = roi_footprint.min()
    if vmax_mask is None:
        vmax_mask = roi_footprint.max()

    # Plot fluorescence image
    im1 = axes[0].imshow(
        fluorescence_image,
        cmap=colormap,
        vmin=vmin_fluo,
        vmax=vmax_fluo,
        interpolation="nearest",
    )
    axes[0].set_title("Fluorescence Image")
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # Plot ROI footprint
    im2 = axes[1].imshow(
        roi_footprint,
        cmap=colormap,
        vmin=vmin_mask,
        vmax=vmax_mask,
        interpolation="nearest",
    )
    axes[1].set_title("ROI Footprint")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    # Plot combined image
    im3 = axes[2].imshow(
        combined_image,
        interpolation="nearest",
    )
    axes[2].set_title("Combined")
    axes[2].axis("off")

    plt.tight_layout()
    return fig
