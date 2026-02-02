"""Streamlit app for labeling red cells."""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Optional

# Add parent directory to path to import cellector
sys.path.insert(0, str(Path(__file__).parent.parent))

from cellector.roi_processor import RoiProcessor
from cellector.io.constructing import create_from_suite2p, create_from_suite3d
from labeling.database import LabelDatabase
from labeling.visualization import get_roi_images, plot_roi_triplet

# Page configuration
st.set_page_config(
    page_title="Red Cell Labeling",
    page_icon="🔴",
    layout="wide",
)

# Initialize session state
if "annotator_name" not in st.session_state:
    st.session_state.annotator_name = None
if "current_roi_id" not in st.session_state:
    st.session_state.current_roi_id = None
if "roi_processor" not in st.session_state:
    st.session_state.roi_processor = None
if "labeled_rois" not in st.session_state:
    st.session_state.labeled_rois = set()
if "db" not in st.session_state:
    st.session_state.db = None


def load_roi_processor(
    data_path: str, data_type: str = "suite2p"
) -> Optional[RoiProcessor]:
    """Load ROI processor from data path.

    Parameters
    ----------
    data_path : str
        Path to the data directory containing ROI processor data.
    data_type : str
        Type of data: "suite2p" or "suite3d". Default is "suite2p".

    Returns
    -------
    Optional[RoiProcessor]
        Loaded ROI processor or None if loading fails.
    """
    try:
        data_path = Path(data_path)
        if not data_path.exists():
            st.error(f"Data path does not exist: {data_path}")
            return None

        if data_type == "suite2p":
            roi_processor = create_from_suite2p(
                data_path,
                autocompute=True,
                use_redcell=True,
            )
        elif data_type == "suite3d":
            roi_processor = create_from_suite3d(
                data_path,
                autocompute=True,
            )
        else:
            st.error(f"Unknown data type: {data_type}. Use 'suite2p' or 'suite3d'.")
            return None

        return roi_processor
    except Exception as e:
        st.error(f"Error loading ROI processor: {e}")
        st.exception(e)
        return None


def get_next_roi_id(
    roi_processor: RoiProcessor, labeled_rois: set, start_from: int = 0
) -> Optional[int]:
    """Get the next unlabeled ROI ID.

    Parameters
    ----------
    roi_processor : RoiProcessor
        The ROI processor instance.
    labeled_rois : set
        Set of ROI IDs that have been labeled.
    start_from : int
        Start searching from this ROI ID.

    Returns
    -------
    Optional[int]
        Next unlabeled ROI ID, or None if all are labeled.
    """
    for roi_id in range(start_from, roi_processor.num_rois):
        if roi_id not in labeled_rois:
            return roi_id
    return None


def main():
    """Main Streamlit app."""
    st.title("🔴 Red Cell Labeling Interface")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # Annotator name input
        annotator_name = st.text_input(
            "Your Name",
            value=st.session_state.annotator_name or "",
            help="Enter your name to identify your labels",
        )

        if annotator_name and annotator_name != st.session_state.annotator_name:
            st.session_state.annotator_name = annotator_name
            # Initialize database
            db_path = Path(__file__).parent / "labels.db"
            st.session_state.db = LabelDatabase(db_path)
            # Load labeled ROIs for this annotator
            st.session_state.labeled_rois = set(
                st.session_state.db.get_labeled_rois(annotator_name)
            )
            st.rerun()

        # Data path input
        st.subheader("Data Loading")
        data_type = st.selectbox(
            "Data Type",
            options=["suite2p", "suite3d"],
            index=0,
            help="Type of data format",
        )
        data_path = st.text_input(
            "Data Path",
            value="",
            help="Path to directory containing ROI processor data",
        )

        if st.button("Load Data") and data_path:
            with st.spinner("Loading ROI processor..."):
                roi_processor = load_roi_processor(data_path, data_type)
                if roi_processor:
                    st.session_state.roi_processor = roi_processor
                    st.success(
                        f"Data loaded successfully! Found {roi_processor.num_rois} ROIs."
                    )
                    st.rerun()

        # Display statistics
        if st.session_state.db and st.session_state.annotator_name:
            st.subheader("Statistics")
            stats = st.session_state.db.get_stats(st.session_state.annotator_name)
            st.metric("Total Labels", stats["total_labels"])
            st.metric("Red Cells", stats["red_count"])
            st.metric("Not Red", stats["not_red_count"])
            st.metric("Unique ROIs", stats["unique_rois"])

    # Main content area
    if not st.session_state.annotator_name:
        st.info("👈 Please enter your name in the sidebar to begin.")
        return

    if st.session_state.roi_processor is None:
        st.info("👈 Please load ROI processor data in the sidebar.")
        return

    roi_processor = st.session_state.roi_processor

    # Get or initialize current ROI
    if st.session_state.current_roi_id is None:
        next_roi = get_next_roi_id(roi_processor, st.session_state.labeled_rois)
        if next_roi is None:
            st.success("🎉 All ROIs have been labeled!")
            return
        st.session_state.current_roi_id = next_roi

    current_roi_id = st.session_state.current_roi_id

    # Display current ROI info
    st.subheader(f"ROI #{current_roi_id}")

    # Check if already labeled
    if st.session_state.db:
        existing_label = st.session_state.db.get_label(
            st.session_state.annotator_name, current_roi_id
        )
        if existing_label is not None:
            label_text = "🔴 Red" if existing_label else "⚪ Not Red"
            st.info(f"Previously labeled as: {label_text}")

    # Image display controls
    col1, col2 = st.columns(2)

    with col1:
        colormap = st.selectbox(
            "Colormap",
            options=["gray", "viridis", "plasma", "inferno", "magma", "hot"],
            index=0,
        )

    with col2:
        auto_scale = st.checkbox("Auto-scale", value=True)

    # Get ROI images
    try:
        fluo_img, mask_img, combined_img = get_roi_images(roi_processor, current_roi_id)

        # Scaling controls
        if not auto_scale:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                vmin_fluo = st.number_input(
                    "Fluo Min",
                    value=float(fluo_img.min()),
                    format="%.2f",
                )
            with col2:
                vmax_fluo = st.number_input(
                    "Fluo Max",
                    value=float(fluo_img.max()),
                    format="%.2f",
                )
            with col3:
                vmin_mask = st.number_input(
                    "Mask Min",
                    value=float(mask_img.min()),
                    format="%.2f",
                )
            with col4:
                vmax_mask = st.number_input(
                    "Mask Max",
                    value=float(mask_img.max()),
                    format="%.2f",
                )
        else:
            vmin_fluo = vmax_fluo = vmin_mask = vmax_mask = None

        # Create and display plot
        fig = plot_roi_triplet(
            fluo_img,
            mask_img,
            combined_img,
            colormap=colormap,
            vmin_fluo=vmin_fluo,
            vmax_fluo=vmax_fluo,
            vmin_mask=vmin_mask,
            vmax_mask=vmax_mask,
        )
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"Error displaying ROI images: {e}")
        st.exception(e)
        return

    # Labeling buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("🔴 Red", type="primary", use_container_width=True):
            if st.session_state.db:
                st.session_state.db.add_label(
                    st.session_state.annotator_name, current_roi_id, True
                )
                st.session_state.labeled_rois.add(current_roi_id)
                st.success("Labeled as Red!")
                # Move to next ROI
                next_roi = get_next_roi_id(
                    roi_processor, st.session_state.labeled_rois, current_roi_id + 1
                )
                st.session_state.current_roi_id = next_roi
                st.rerun()

    with col2:
        if st.button("⚪ Not Red", type="secondary", use_container_width=True):
            if st.session_state.db:
                st.session_state.db.add_label(
                    st.session_state.annotator_name, current_roi_id, False
                )
                st.session_state.labeled_rois.add(current_roi_id)
                st.success("Labeled as Not Red!")
                # Move to next ROI
                next_roi = get_next_roi_id(
                    roi_processor, st.session_state.labeled_rois, current_roi_id + 1
                )
                st.session_state.current_roi_id = next_roi
                st.rerun()

    with col3:
        if st.button("⏭️ Skip", use_container_width=True):
            # Move to next ROI without labeling
            next_roi = get_next_roi_id(
                roi_processor, st.session_state.labeled_rois, current_roi_id + 1
            )
            st.session_state.current_roi_id = next_roi
            st.rerun()

    # Navigation controls
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("⏮️ First"):
            first_roi = get_next_roi_id(roi_processor, set(), 0)
            if first_roi is not None:
                st.session_state.current_roi_id = first_roi
                st.rerun()

    with col2:
        if st.button("◀️ Previous"):
            prev_roi = current_roi_id - 1
            if prev_roi >= 0:
                st.session_state.current_roi_id = prev_roi
                st.rerun()

    with col3:
        if st.button("Next ▶️"):
            next_roi = current_roi_id + 1
            if next_roi < roi_processor.num_rois:
                st.session_state.current_roi_id = next_roi
                st.rerun()

    with col4:
        if st.button("Last ⏭️"):
            # Find last unlabeled ROI
            for roi_id in range(roi_processor.num_rois - 1, -1, -1):
                if roi_id not in st.session_state.labeled_rois:
                    st.session_state.current_roi_id = roi_id
                    st.rerun()
                    break


if __name__ == "__main__":
    main()
