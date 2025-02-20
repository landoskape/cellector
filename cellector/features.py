from typing import List
from .utils import (
    phase_correlation_zero,
    dot_product,
    compute_correlation,
    in_vs_out,
    surround_filter,
)
from .filters import filter


class FeaturePipeline:
    """
    Pipeline that defines a feature computation method and its dependencies on attributes of roi_processor instances.

    Attributes
    ----------
    name : str
        Name of the feature pipeline.
    method : callable
        Method that computes the feature, accepting an roi_processor instance as an input and returns a np.ndarray
        which associates each ROI with a feature value.
    dependencies : List[str]
        List of attributes of roi_processor that the feature computation method depends on.
    """

    def __init__(self, name: str, method: callable, dependencies: List[str]):
        self.name = name
        self.method = method
        self.dependencies = dependencies


def compute_phase_correlation(roi_processor):
    """Compute the phase correlation between the masks and reference images.

    Returns
    -------
    np.ndarray
        The phase correlation between the masks and reference images across planes.

    See Also
    --------
    utils.phase_correlation_zero : Function that computes the phase correlation values.
    """
    # Input to phase correlation is centered masks and centered references
    centered_masks = roi_processor.centered_masks
    centered_references = roi_processor.centered_references

    # Window the centered masks and references
    windowed_masks = filter(
        centered_masks, "window", kernel=roi_processor.parameters["window_kernel"]
    )
    windowed_references = filter(
        centered_references, "window", kernel=roi_processor.parameters["window_kernel"]
    )

    # Phase correlation requires windowing
    return phase_correlation_zero(
        windowed_masks,
        windowed_references,
        eps=roi_processor.parameters["phase_corr_eps"],
    )


def compute_dot_product(roi_processor):
    """Compute the dot product between the masks and filtered reference images.

    Returns
    -------
    np.ndarray
        The dot product between the masks and reference images across planes, normalized
        by the norm of the mask intensity values.

    See Also
    --------
    utils.dot_product : Function that computes the dot product values.
    utils.dot_product_array : Alternative that uses mask images instead of weights and pixel indices.
    """
    lam = roi_processor.lam
    ypix = roi_processor.ypix
    xpix = roi_processor.xpix
    plane_idx = roi_processor.plane_idx
    filtered_references = roi_processor.filtered_references
    return dot_product(lam, ypix, xpix, plane_idx, filtered_references)


def compute_corr_coef(roi_processor):
    """Compute the correlation coefficient between the masks and reference images.

    Returns
    -------
    np.ndarray
        The correlation coefficient between the masks and reference images across planes.

    See Also
    --------
    utils.compute_correlation : Function that computes the correlation coefficient values.
    """
    centered_masks = roi_processor.centered_masks
    filtered_centered_references = roi_processor.filtered_centered_references
    iterations = roi_processor.parameters["surround_iterations"]
    masks_surround, references_surround = surround_filter(
        centered_masks, filtered_centered_references, iterations=iterations
    )
    return compute_correlation(masks_surround, references_surround)


def compute_in_vs_out(roi_processor):
    """Compute the in vs. out feature for each ROI.

    The in vs. out feature is the ratio of the dot product of the mask and reference
    image inside the mask to the dot product inside plus outside the mask.

    Returns
    -------
    np.ndarray
        The in vs. out feature for each ROI.

    See Also
    --------
    utils.in_vs_out : Function that computes the in vs. out feature values.
    """
    centered_masks = roi_processor.centered_masks
    centered_references = roi_processor.centered_references
    iterations = roi_processor.parameters["surround_iterations"]
    return in_vs_out(centered_masks, centered_references, iterations=iterations)


# Mapping of feature pipelines to their corresponding methods
PIPELINE_METHODS = dict(
    phase_corr=compute_phase_correlation,
    dot_product=compute_dot_product,
    corr_coef=compute_corr_coef,
    in_vs_out=compute_in_vs_out,
)

# Mapping of feature pipelines to dependencies on attributes of roi_processor instances
PIPELINE_DEPENDENCIES = dict(
    phase_corr=["centered_width", "centroid_method", "window_kernel", "phase_corr_eps"],
    dot_product=["lowcut", "highcut", "order"],
    corr_coef=[
        "surround_iterations",
        "centered_width",
        "centroid_method",
        "lowcut",
        "highcut",
        "order",
    ],
    in_vs_out=["surround_iterations", "centered_width", "centroid_method"],
)

# Create a list of standard pipelines
standard_pipelines = []
for name, method in PIPELINE_METHODS.items():
    dependencies = PIPELINE_DEPENDENCIES[name]
    pipeline = FeaturePipeline(name, method, dependencies)
    standard_pipelines.append(pipeline)
