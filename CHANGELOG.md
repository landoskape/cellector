
# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
 
## [0.1.1] - 2024-11-19

### Added
Saving features is now optional! The create_from_{...} functions now have an optional
input argument called ``save_features`` that is passed to the ``RoiProcessor``. This
determines if feature values are saved to disk automatically. The default value is True,
but you might want to set it to False for your purposes. 

### Changed
Removed the "Control-c" key command for saving. You can save by clicking the button.

### Fixed
Updated maximum python version - some dependencies are not compatible with python 3.13 yet.