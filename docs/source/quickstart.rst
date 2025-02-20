Quickstart
==========

Getting Started with Cellector
------------------------------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install cellector

Basic Usage
~~~~~~~~~~~

Here's a minimal example to get you started:

.. code-block:: python

   import cellector
   from cellector import RoiProcessor, CellectorManager

   # Create a RoiProcessor instance from suite2p output
   roi_processor = cellector.io.create_from_suite2p("path/to/suite2p/dir")

   # Create a CellectorManager instance
   manager = CellectorManager.make_from_roi_processor(roi_processor)

   # Launch the GUI
   from cellector.gui import SelectionGUI
   gui = SelectionGUI(roi_processor) 