
# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
 
## [upcoming] - 

The gui class is a beast. It needs to be broken down into components. For example, there
should be a method for scripting the application of criteria to feature values, but right
now the only way to do this is to open the GUI for each session... So that part and related
components should be removed from the GUI to an independent module which can be accessed by
the GUI or by other methods for scripting. 

Let's change to "selection" and "focus" so idx_selection means the ROIs selected by feature
criteria and manual annotation. Focus is what's currently shown right now.

## [0.2.0] - YYYY-MM-DD --- NOT UPLOADED YET

### Added
Saving features is now optional! The create_from_{...} functions now have an optional
input argument called ``save_features`` that is passed to the ``RoiProcessor``. This
determines if feature values are saved to disk automatically. The default value is True,
but you might want to set it to False for your purposes. 

Added more functions for determining paths for consistency and using the DRY principle. 

### Changed
#### Major change: filepath structure for features
The structure of filepaths for features and feature criteria have been changed to
{feature_name}_feature.npy and {feature_name}_featurecriteria.npy. The reason for this
change is so that it's possible to determine which features and criteria have been saved
by inspecting filenames (whereas before only criteria was immediately identifiable). This
will cause backwards incompatibility because files on the old path will not be
recognized. To address this change, two supporting methods are provided called 
``identify_cellector_folders`` and ``update_feature_paths``. You can use ``identify...``
to get all folders that contain a cellector directory and ``update...`` to convert the
filepaths to the new structure. These functions are in cellector/io/operations. 
```python
from pathlib import Path
from cellector.io.operations import identify_cellector_folders, update_feature_paths
top_level_dir = Path(r"C:\Users\Andrew\Documents")
cellector_folders = identify_cellector_folders(top_level_dir)
update_feature_paths(cellector_folders)
```

#### Minor changes: 
Removed the "Control-c" key command for saving. You can save by clicking the button.
The IO module is broken down into a directory and is more organized. 

### Fixed
Updated maximum python version - some dependencies are not compatible with python 3.13 yet.