# cellector
[![PyPI version](https://badge.fury.io/py/cellector.svg)](https://badge.fury.io/py/cellector)
[![Documentation Status](https://readthedocs.org/projects/cellector/badge/?version=stable)](https://cellector.readthedocs.io/en/stable/?badge=stable)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<!-- [![Tests](https://github.com/landoskape/cellector/actions/workflows/tests.yml/badge.svg)](https://github.com/landoskape/cellector/actions/workflows/tests.yml) -->
<!-- [![codecov](https://codecov.io/gh/landoskape/cellector/branch/main/graph/badge.svg)](https://codecov.io/gh/landoskape/cellector) -->

A pipeline and GUI for determining which ROIs match features in a fluorescence image. It
is a common challenge in biology to determine whether a particular ROI (i.e. a collection
of weighted pixels representing an inferred structure in an image) overlaps with features
of a fluorescence image co-registered to the ROI. For example, in neuroscience, we might
use [suite2p](https://github.com/MouseLand/suite2p) to extract ROIs indicating active
cells using a functional fluorophore like GCaMP, but want to know if the cells associated
with those ROIs contain a secondary fluorophore like tdTomato. This package helps you do
just that!

The package itself is somewhat simple, but we spent lots of time thinking about how to do
this in the most reliable way. The standard pipeline computes a set of standard features
for each ROI in comparison to a reference image which are useful for determining whether
an ROI maps onto fluorescence. We provide a GUI for viewing the ROIs, the reference
images, a distribution of feature values for each ROI, and an interactive system for
deciding where to draw cutoffs on each feature to choose the ROIs that contain
fluorescence. There's also a system for manual annotation if the automated system doesn't
quite get it all right. 

## Installation
This package is installable with pip from PyPI. It is a lightweight package with minimal
dependencies, so is probably compatible with other python environments you might use. 
If you're starting from scratch, first make a python environment, activate it, and
install ``cellector`` with pip. If you are using an existing environment, skip the first
two steps and just do pip install from within the environment. 
```bash
conda create -n cellector
conda activate cellector
pip install cellector
```

## Usage and Tutorial
The basic workflow of this package is as follows:
1. Construct an ``RoiProcessor`` object.
2. Use the ``SelectionGUI``. 
3. Save the data.
4. Repeat (or use scripting to speed up).

If you want to see the basic workflow in a notebook, look [here](https://github.com/landoskape/cellector/blob/main/notebooks/tutorial.ipynb).
Otherwise, read the instructions below or look at the [documentation](https://cellector.readthedocs.io/). 

### Basic instructions
We've provided a few functions to make ``RoiProcessor`` objects that work differently
depending on what kind of data you are starting with. For an exhaustive list, check out
the explanation in the documentation files [here](https://cellector.readthedocs.io/en/latest/examples.html).

If you are working directly on the output of suite2p, use:
```python
from cellector.io import create_from_suite2p
suite2p_dir = # define your suite2p path - the one with plane0, plane1, ... in it
roi_processor = create_from_suite2p(suite2p_dir)
```

Then, open the ``SelectionGUI`` as follows:
```python
from cellector.gui import SelectionGUI
gui = SelectionGUI(roi_processor)
```

Then, use the GUI and hit save! Instructions for the GUI are [here](https://cellector.readthedocs.io/en/latest/gui.html).

### Scripting
The GUI works, but it can be a bit tedious to open it over and over again when you know
you want the same settings for a group of sessions. To enable quick application of 
feature criteria settings to many sessions, we have included scripting tools. 

```python
from cellector.io import propgate_criteria
from cellector.manager import CellectorManager

# Copy criteria from suite2p_dir to all the other directories
other_directories = [Path(r"C:\Path\to\other\suite2p"), Path(r"C:\Path\to\another\suite2"), ...] # as many as you like
success, failure = propagate_criteria(suite2p_dir, *other_directories)

for directory in other_directories:
    # Make an roi_processor for each session(directory), this will compute features and save the data
    roi_processor = create_from_suite2p(directory) # or whichever method you used to create the roi_processor
    
    # Make a manager instance
    manager = CellectorManager.make_from_roi_processor(roi_processor)
    
    # this will save the updated criteria and idx_selection to cellector directory
    # it will also save empty manual label arrays if they don't exist
    manager.save_all() 
```

### Handling convention changes in new versions
Several changes will prevent or complicate backwards compatibility. Here's what you need
to know:
- `targetcell.npy` → `idx_selection.npy`: the name convention of the main output has been changed from targetcell.npy to idx_selection.npy
- Manual selection shape from `(num_rois, 2)` to `(2, num_rois)`: manual selection is changed from a stack across labels and active_label
- Feature files from `{feature_name}.npy` to `{feature_name}_feature.npy`: feature files now have a suffix for automatic identification
- Criteria files from `{feature_name}_criteria.npy` to `{feature_name}_featurecriteria.npy`: criteria files suffix changed

To address these changes, version 1.0.0 includes migration utilities to fix existing data
structures. You can use ``identify_celector_folders`` to get all folders that contain a
cellector directory. The other three functions operate on this list and fix the filenames
or data structure (transposing manual selection) on all cellector files. These functions
are in cellector/io/operations. 
```python
from pathlib import Path
from cellector.io import identify_cellector_folders
from cellector.io import update_feature_paths
from cellector.io import update_manual_selection_shape
from cellector.io import update_idx_selection_filenames
top_level_dir = Path(r"C:\some\path\that\has\all\the\cellector\directories\beneath\it")
root_dirs = identify_cellector_folders(top_level_dir)
update_idx_selection_filenames(root_dirs)
update_manual_selection_shape(root_dirs)
update_feature_paths(root_dirs)
```

Note that a few other things have changed in version 1.0.0, see the [CHANGELOG](https://github.com/landoskape/cellector/blob/main/CHANGELOG.md)
for more detailed descriptions!

## Features in Progress
### Hyperparameter Choices
There are a few "hyperparameters" to the package, including filtering parameters, the eps
value for phase correlation, and size parameters for centered stacks. We need to enable 
hyperparameter optimization for these, which a user can supervise themselves. Idea:
The user could open a GUI that compares masks with reference images for some sample
"true" data and in addition for any data they've loaded in. One idea:
For a particular set of hyperparameters (filtering, for example), the user could get a
histogram of feature values for all the features for all the masks. They could use cutoff
lines to pick a range of feature values for that particular set of hyperparameters, and
then scroll through mask matches that come from within that range. This way, they could
determine how the hyperparameters affect the feature values at each part of the
distribution and select hyperparameters that give good separation.
In addition, there could be some automated tuning, for example, to pick the eps a user
could just input the maximum size ROI, and then measuring the average power for higher
spatial frequencies.

### Visualization of algorithm and filtering steps etc
To help choose hyperparameters and see how it's working, I'm going to build some tools to
visualize masks and the reference image under different conditions. 

## Contributing
Feel free to contribute to this project by opening issues or submitting pull
requests. It's already a collaborative project, so more minds are great if you
have ideas or anything to contribute!

## License & Citations
This project is licensed under the GNU License. If you use this repository as part of a
publication, please cite us. There's no paper associated with the code at the moment, but
you can cite our GitHub repository URL or email us for any updates about this issue.
