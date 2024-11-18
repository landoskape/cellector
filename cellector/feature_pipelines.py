from typing import List
from .features import compute_phase_correlation, compute_dot_product, compute_corr_coef, compute_in_vs_out


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
    corr_coef=["surround_iterations", "centered_width", "centroid_method", "lowcut", "highcut", "order"],
    in_vs_out=["surround_iterations", "centered_width", "centroid_method"],
)

# Create a list of standard pipelines
standard_pipelines = []
for name, method in PIPELINE_METHODS.items():
    dependencies = PIPELINE_DEPENDENCIES[name]
    pipeline = FeaturePipeline(name, method, dependencies)
    standard_pipelines.append(pipeline)
