# red-cell-selector
 A pipeline and GUI for determining which ROIs are red

# Filtering
There's a lot of filtering that happens. I think it would be useful to build a central
filtering library with the typical options and easily extendable filtering methods, but 
you know this could be more trouble than it's worth. I guess the idea is as follows:
1. Build a filtering library that is informed by a dictionary what to do. For example, the
dict might look like this:
```python
instructions = dict(
    name="window",
    method="hanning",
)
```
Which contains the name (window) and the method (hanning) and that's all that's needed to 
window an image. Then, if a new method is added, the string could change but the calling
function would stay the same (``filter(image, **instructions)``). 
2. Clearly identify what the "standard" filtering operations are for each feature pipeline, 
and maybe even include ways for the user to study different filtering options in a GUI? 

### Filtering Use in Original Module
1. Correlation coefficient:
 - butterworthbpf filter on reference stack before centering: width=20, lowcut=12, highcut=250, order=3, fs=512
 - nan fill (outside the width on masks and reference)
2. Dot Product:
 - butterworthbpf filter on reference stack: lowcut=12, highcut=250, order=3, fs=512
3. Phase Correlation
 - no filtering, width=40, eps=1e6, winFunc=hamming
 - window on both reference and masks

# Hyperparameter Choices:
So far the only hyperparameters I'm aware of are filtering parameters, and the eps value for
phase correlation measurements (which is weirdly high...?). I think it would be good to do 
some hyperparameter optimization for these, which a user could manually supervise themselves
with some labelling. For example, the user could open a GUI that compares masks with reference
images for some sample "true" data and in addition for any data they've loaded in. The GUI might
look something like follows:
1. For a particular set of hyperparameters (filtering, for example), the user could get a histogram
of feature values for all the features for all the masks. They could use cutoff lines to pick a range
of feature values for that particular set of hyperparameters, and then scroll through mask matches
that come from within that range. This way, they could determine how the hyperparameters affect the
feature values at each part of the distribution and select hyperparameters that give good separation.

# GUI
ToDo: study the GUI code to figure out what needs to be processed / what needs to be quickly
accessible, etc etc