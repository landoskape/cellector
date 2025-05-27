from typing import List
from functools import partial
import numpy as np
from .utils import (
    phase_correlation_zero,
    dot_product,
    compute_correlation,
    in_vs_out,
    surround_filter,
)
from .filters import filter

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .roi_processor import RoiProcessor


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


def compute_phase_correlation(
    roi_processor: "RoiProcessor",
    functional: bool = False,
) -> np.ndarray:
    """Compute the phase correlation between the masks and reference images.

    Parameters
    ----------
    roi_processor : RoiProcessor
        The roi_processor instance to compute the phase correlation for.
    functional : bool, optional
        Whether to compute the phase correlation between the masks and functional reference images.
        Default is False.

    Returns
    -------
    np.ndarray
        The phase correlation between the masks and reference images across planes.

    See Also
    --------
    utils.phase_correlation_zero : Function that computes the phase correlation values.
    """
    # Input to phase correlation is centered masks and centered reference
    centered_masks = roi_processor.centered_masks
    if functional:
        centered_reference = roi_processor.centered_reference_functional
    else:
        centered_reference = roi_processor.centered_reference

    # Window the centered masks and reference
    windowed_masks = filter(
        centered_masks, "window", kernel=roi_processor.parameters["window_kernel"]
    )
    windowed_reference = filter(
        centered_reference, "window", kernel=roi_processor.parameters["window_kernel"]
    )

    # Phase correlation requires windowing
    return phase_correlation_zero(
        windowed_masks,
        windowed_reference,
        eps=roi_processor.parameters["phase_corr_eps"],
        volumetric=roi_processor.volumetric,
    )


def compute_dot_product(
    roi_processor: "RoiProcessor",
    functional: bool = False,
) -> np.ndarray:
    """Compute the dot product between the masks and filtered reference images.

    Parameters
    ----------
    roi_processor : RoiProcessor
        The roi_processor instance to compute the dot product for.
    functional : bool, optional
        Whether to compute the dot product between the masks and functional reference images.
        Default is False.

    Returns
    -------
    np.ndarray
        The dot product between the masks and reference images across planes, normalized
        by the norm of the mask intensity values.

    See Also
    --------
    utils.dot_product : Function that computes the dot product values.
    """
    lam = roi_processor.lam
    ypix = roi_processor.ypix
    xpix = roi_processor.xpix
    zpix = roi_processor.zpix
    if functional:
        filtered_reference = roi_processor.filtered_reference_functional
    else:
        filtered_reference = roi_processor.filtered_reference
    return dot_product(
        lam,
        ypix,
        xpix,
        zpix,
        filtered_reference,
        volumetric=roi_processor.volumetric,
    )


def compute_corr_coef(
    roi_processor: "RoiProcessor",
    functional: bool = False,
) -> np.ndarray:
    """Compute the correlation coefficient between the masks and reference images.

    Parameters
    ----------
    roi_processor : RoiProcessor
        The roi_processor instance to compute the correlation coefficient for.
    functional : bool, optional
        Whether to compute the correlation coefficient between the masks and functional reference images.
        Default is False.

    Returns
    -------
    np.ndarray
        The correlation coefficient between the masks and reference images across planes.

    See Also
    --------
    utils.compute_correlation : Function that computes the correlation coefficient values.
    """
    centered_masks = roi_processor.centered_masks
    if functional:
        filtered_centered_reference = (
            roi_processor.filtered_centered_reference_functional
        )
    else:
        filtered_centered_reference = roi_processor.filtered_centered_reference
    iterations = roi_processor.parameters["surround_iterations"]
    masks_surround, reference_surround = surround_filter(
        centered_masks,
        filtered_centered_reference,
        iterations=iterations,
        volumetric=roi_processor.volumetric,
    )
    return compute_correlation(
        masks_surround,
        reference_surround,
        volumetric=roi_processor.volumetric,
    )


def compute_in_vs_out(
    roi_processor: "RoiProcessor",
    functional: bool = False,
) -> np.ndarray:
    """Compute the in vs. out feature for each ROI.

    The in vs. out feature is the ratio of the dot product of the mask and reference
    image inside the mask to the dot product inside plus outside the mask.

    Parameters
    ----------
    roi_processor : RoiProcessor
        The roi_processor instance to compute the in vs. out feature for.
    functional : bool, optional
        Whether to compute the in vs. out feature between the masks and functional reference images.
        Default is False.

    Returns
    -------
    np.ndarray
        The in vs. out feature for each ROI.

    See Also
    --------
    utils.in_vs_out : Function that computes the in vs. out feature values.
    """
    centered_masks = roi_processor.centered_masks
    if functional:
        centered_reference = roi_processor.centered_reference_functional
    else:
        centered_reference = roi_processor.centered_reference
    iterations = roi_processor.parameters["surround_iterations"]
    return in_vs_out(
        centered_masks,
        centered_reference,
        iterations=iterations,
        volumetric=roi_processor.volumetric,
    )


# Mapping of feature pipelines to their corresponding methods
PIPELINE_METHODS = dict(
    phase_corr=compute_phase_correlation,
    dot_product=compute_dot_product,
    corr_coef=compute_corr_coef,
    in_vs_out=compute_in_vs_out,
)

# Functional features are really just there to check if the red signal is
# coming from background green signal (e.g. bright highly active cells). So
# right now we only need the dot product pipeline which emphasizes this
# aspect of the functional reference image.
FUNCTIONAL_PIPELINE_METHODS = dict(
    dot_product=compute_dot_product,
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

# Create a list of functional pipelines
functional_pipelines = []
for name, method in FUNCTIONAL_PIPELINE_METHODS.items():
    dependencies = PIPELINE_DEPENDENCIES[name]
    method_on_functional = partial(method, functional=True)
    pipeline = FeaturePipeline("functional_" + name, method_on_functional, dependencies)
    functional_pipelines.append(pipeline)
