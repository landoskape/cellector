# Red Cell Labeling Interface

A Streamlit web interface for collecting human-labeled datasets of red cells.

## Features

- **Simple Interface**: Clean, intuitive UI for labeling cells as "Red" or "Not Red"
- **Image Display**: Shows fluorescence image, ROI footprint, and combined overlay in a (1, 3) subplot layout
- **Customizable Visualization**: Adjustable colormaps and scaling controls
- **Database Storage**: SQLite database stores all labels with timestamps and annotator names
- **Progress Tracking**: Statistics showing labeling progress
- **Navigation**: Easy navigation between ROIs with skip, previous, and next buttons

## Installation

Install the required dependencies:

```bash
pip install streamlit
```

The app uses the cellector package which should already be installed in your environment.

## Usage

### Starting the App

Run the Streamlit app:

```bash
streamlit run labeling/app.py
```

The app will open in your web browser.

### Loading Data

1. Enter your name in the sidebar (this identifies your labels)
2. Select the data type (suite2p or suite3d)
3. Enter the path to your data directory
4. Click "Load Data"

For suite2p data, the directory should contain:
- `plane0/`, `plane1/`, etc. folders, each with:
  - `stat.npy`
  - `ops.npy`
  - `redcell.npy` (optional)

For suite3d data, the directory should contain:
- `stats.npy`
- `ref_img_3d_structural.npy`
- `ref_img_3d.npy`

### Labeling Cells

1. Review the displayed images (fluorescence, ROI footprint, and combined)
2. Adjust colormap and scaling if needed
3. Click "🔴 Red" or "⚪ Not Red" to label the current ROI
4. The app automatically moves to the next unlabeled ROI

### Navigation

- **Skip**: Move to next ROI without labeling
- **Previous/Next**: Navigate manually through ROIs
- **First/Last**: Jump to first or last unlabeled ROI

### Database

Labels are stored in `labeling/labels.db` (SQLite database). Each label includes:
- Annotator name
- ROI ID
- Label (Red/Not Red)
- Timestamp

## Statistics

The sidebar displays:
- Total labels you've made
- Number of red cells labeled
- Number of not-red cells labeled
- Number of unique ROIs labeled

## Notes

- Labels are saved immediately when you click a button
- You can re-label ROIs (previous labels will be updated)
- The app remembers which ROIs you've labeled and skips them when moving forward
- For volumetric data (suite3d), images are summed across planes for display
