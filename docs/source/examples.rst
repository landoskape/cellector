Examples
========

This is woefully incomplete. Bug me to add more!

.. code-block:: python

    from cellector.io import create_from_mask_volume, create_from_pixel_data
    root_dir = # folder to save results to
    stats = # List of dictionaries containing mask data
    mask_volume = # 3D array of stacked mask images - one for each ROI (slower alternative to stats)
    reference_images = # 3D array of reference images - one for each plane
    plane_idx = # 1D array relating each mask to the apppropriate reference image

    # if you have stats already, use:
    roi_processor = create_from_pixel_data(root_dir, stats, reference_images, plane_idx)
    # or if you're starting from a mask_volume:
    roi_processor = create_from_mask_volume(root_dir, mask_volume, reference_images, plane_idx)