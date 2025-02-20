GUI Documentation
=================

How to use the GUI
------------------
This is a short explanation of how to use the GUI. There will be more explanations later,
but for now this is just to get people started with the basic functionality. 

.. image:: ../media/full_gui.png
   :alt: full-gui

Components
----------
The GUI has several components, indicated on the image displayed above. 

Napari viewer
~~~~~~~~~~~~~
This is a standard napari app (you can look at the documentation for it on napari's 
website). There are three layers displayed:

1. Mask images, which are a rendering of the mask intensity data - e.g. "lam" from suite2p
2. Mask labels, which are a flat color rendering for each mask. In random color mode, each
   mask get assigned a random color. In feature mapping modes (try pressing the "c" key),
   they map each label to it's value in one of the features displayed below. The colormap can
   be changed with the colormap button (or by pressing the "a" key). 
3. Reference image, which is a graymap intensity image of the reference fluorescence. You
   can turn this on or off by pressing the "r" key. 

You can turn off the masks (images or labels) by pressing the "v" key. 
You can turn off the reference by pressing the "r" key. 
You can switch between mask images and labels by pressing the "s" key. 
You can switch between displaying selected ROIs and control ROIs with the "t" key. 

The slider below the viewer area controls which plane is being observed.

Information Bar
~~~~~~~~~~~~~~~
Immediately below the viewer is a little text area which displays useful information and
feedback to the user. 

Feature Cutoff Toggles
~~~~~~~~~~~~~~~~~~~~~~
Each feature is associated with a minimum and maximum cutoff (the two vertical yellow
lines on each histogram). You can turn on or off the cutoff by pressing the button above
each histogram. The text on the button indicates whether the cutoff is being used. If a
cutoff isn't used, then the saved value will be ``None``.

Feature histograms
~~~~~~~~~~~~~~~~~~
These show the distribution of feature values for all the ROIs in the current plane for 
each feature. The name of the feature is above the histogram. The full distribution is 
shown in gray, and the currently selected ROIs is shown in red. 

Buttons
~~~~~~~
There are a few buttons which control the GUI. They're all explained below. Some of them
are controlled by key strokes. 

Key strokes
-----------
t
  toggle control vs target cells in selection

s
  switch between mask images and mask labels

v
  turn on/off the visibility of masks

r
  turn on/off the visibility of the reference

c
  go to next color state (i.e. random, or pseudocoloring each label by its feature value)

a
  change colormap (only applies if not in random color state)

control-c
  save selection (this will save the target cells - not the currently selected 
  cells, so if you are currently showing control cells, it'll always save control cells).

Buttons
-------
save_selection
  saves the target cells and all the data (features, feature cutoffs "criteria", 
  target idx). Note, this will always save the target cells, not the currently selected cells,
  so if you are currently showing control cells, it'll still just show the target ones. 

target cells / control cells
  the second button switches between showing target and control cells.

"using manual labels"
  the third button switches between using or ignoring manual labels. 

"all labels"
  the fourth button determines whether to use all ROIs or just show manual labels.

"clear manual labels"
  turns off the use of all manual labels for all the ROIs. You have to be
  holding "control" down to get this to work because it is irreversible once you save and overwrite.

"random"
  the 6th button determines how to color ROIs - random is a random color, and any other
  choice will pseudo color them based on their feature value for the selected feature. 

"plasma"
  the last button determines which colormap to use. 

Operation
---------
I generally like to pick one feature that is best for picking cells (it's almost always the
phase correlation!!!!). So, I start by moving up the minimum cutoff until all the cells look
like they match the background fluorescence to me. It's helpful to switch between looking at
mask images and labels (key command "s"), or turning off masks back and forth (with "v") to 
get a clear sense of how each mask maps onto the fluorescence. You can zoom in to particular 
masks to see. 

Once I get a decent minimum cutoff, I refine it as follows. I bring the maximum line below 
the minimum cutoff point (they'll switch roles), and look at the few cells that meet this
narrow range. If they look like they clearly match fluorescence, I go lower. The idea is to
figure out where in the distribution it stops mapping onto fluorescence. Once it's set, I then
keep the minimum where I want it, and turn off the maximum with the toggle above the feature 
histogram (it'll turn red and say "ignore..."). 

It can be helpful to click on the ROI (in label mode, use "s") and the text
area will display the masks's feature values for you to hone in on the best cutoff. 

There might be some edge cases - where the ideal cutoff line doesn't quite divide cells. We 
have two systems for this: using other features and manual annotation. 

You can then do the same thing with other features as described above, and just figure out if
another feature can help divide the ROIs appropriately. 

If that doesn't work, use manual annotation. If you double click on an ROI, it'll be manually
annotated and will flip assignment (that is, if you are currently showing target cells, it'll 
be annotated control, and vice versa). You can only do this in label mode (try "s"). You can 
also clear a manual annotation if you don't like it. You have to be in label mode, you have to
be looking at the ROI, and you can only be showing manual labels (click the "all labels" button
at the bottom). Then, if you control-click an ROI, it's manual annotation will be removed. You
can also clear all manual labels with the button on the bottom. 

The saved idx_target array will first use feature cutoffs, then any active manual annotations 
will overwrite for each ROI. 

Once you're done, press control-c or press the save selection button!
