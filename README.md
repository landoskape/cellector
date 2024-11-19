# cellector
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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

If you want to see the basic workflow in a notebook, look [here](./notebooks/tutorial.ipynb).
Otherwise, read the instructions below or look at the [documentation](./docs/examples.md)
which is very far from complete at the moment. 

#### Basic instructions
We've provided a few functions to make ``RoiProcessor`` objects that work differently
depending on what kind of data you are starting with. For an exhaustive list, go
[here](./docs/examples.md). If you are working directly on the output of suite2p, use:
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

Then, use the GUI and hit save! Instructions for the GUI are coming. 

## Contributing
Feel free to contribute to this project by opening issues or submitting pull
requests. It's already a collaborative project, so more minds are great if you
have ideas or anything to contribute!

## License & Citations
This project is licensed under the GNU License. If you use this repository as part of a
publication, please cite us. There's no paper associated with the code at the moment, but
you can cite our GitHub repository URL or email us for any updates about this issue.
