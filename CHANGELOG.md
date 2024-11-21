
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

cellector folders contain npy files of features, criteria, selection, and manual annotation.
It is probably possible to identify all features by looking for files that end in *_criteria.npy,
but maybe there should be another file called "features" that contains a list of stored features?

## [0.1.1] - YYYY-MM-DD --- NOT UPLOADED YET

### Added
Saving features is now optional! The create_from_{...} functions now have an optional
input argument called ``save_features`` that is passed to the ``RoiProcessor``. This
determines if feature values are saved to disk automatically. The default value is True,
but you might want to set it to False for your purposes. 

Added *_path functions for consistency and DRYing in the IO module.

### Changed
Removed the "Control-c" key command for saving. You can save by clicking the button.

### Fixed
Updated maximum python version - some dependencies are not compatible with python 3.13 yet.