How It Works
============

Core Concepts
-------------

Cellector is designed to help you select cells based on their features in reference to fluorescence images. Here's how the main components work together:

ROI Processing
~~~~~~~~~~~~~~

The RoiProcessor class handles:

1. Loading and organizing mask & fluorescence data
2. Computing features for each ROI
3. Managing data across multiple image planes

Feature Management
~~~~~~~~~~~~~~~~~~

The CellectorManager class provides:

1. Feature storage and criteria management
2. Manual and automated cell selection
3. Persistence of selection criteria 

Standard ROI Quality Features
-----------------------------

The cellector package computes four standard features to assess ROI quality:

Phase Correlation
~~~~~~~~~~~~~~~~~
This feature measures how well the spatial pattern of the ROI mask matches the reference image using phase correlation in the Fourier domain. See :func:`cellector.utils.phase_correlation_zero` for implementation details.

The computation:

1. Windows both the mask and reference images
2. Computes the normalized cross-power spectrum:

.. math::

   R = \frac{F(mask) \cdot \text{conj}(F(reference))}{\epsilon + |F(mask) \cdot \text{conj}(F(reference))|}

3. Returns the real component at zero offset, which indicates pattern similarity without any shifts

Dot Product
~~~~~~~~~~~
This feature measures how well the ROI mask aligns with bright regions in the reference image. See :func:`cellector.utils.dot_product` for implementation details.

For each ROI with intensity values λ at pixels (y,x):

.. math::

   \text{dot_product} = \frac{\sum_i \lambda_i \cdot \text{reference}(y_i,x_i)}{\|\lambda\|}

The normalization by the mask intensity norm makes this measure independent of the overall brightness of the ROI.

Correlation Coefficient
~~~~~~~~~~~~~~~~~~~~~~~
This computes the Pearson correlation coefficient between the mask and reference image in the local region around the ROI. See :func:`cellector.utils.compute_correlation` for implementation details.

.. math::

   \text{corr} = \frac{1}{N} \sum \frac{(mask - \mu_{mask})(reference - \mu_{ref})}{\sigma_{mask}\sigma_{ref}}

The computation:

1. Applies a surround filter to focus on the local region near the ROI

.. note::
    The surround region is a shape-fitting mask around the ROI. Everything in this region will be set to 0 for the ROI. Everything outside the ROI's surround will be set to 1 for both the mask and the reference. This means that the computation is comparing the local structure of the ROI with the local structure of the reference image -- on a scale set by how different the center region is from the surround in the reference image. 

2. Removes the mean of both mask and reference
3. Normalizes by standard deviations
4. Averages over the valid pixels (N)

In vs Out
~~~~~~~~~
This feature compares the reference image intensity inside the ROI to its surrounding region. See :func:`cellector.utils.in_vs_out` for implementation details.

.. math::

   \text{in_vs_out} = \frac{\sum_{inside} reference}{\sum_{inside} reference + \sum_{surround} reference}

Where:

- "inside" is the ROI mask footprint
- "surround" is defined by dilating the ROI mask (default 7 iterations)

This is similar to Suite2p's red cell probability feature and helps identify ROIs that capture distinct features in the reference image.

Red S2P
~~~~~~~
If you are working from suite2p data, you will also see "red_s2p" as a feature. This is the red cell probability computed by suite2p, which is very similar to the in vs out feature, but instead of using a surround filter based on pixels, it uses a surround filter for "out" based on the neuropil mask. 
