from typing import List, Optional, Dict, Union
from pathlib import Path
from copy import deepcopy
import numpy as np
from . import utils
from .filters import filter
from .features import FeaturePipeline, standard_pipelines, functional_pipelines

# Might be useful for optimizing parameters
# from sklearn.model_selection import ParameterGrid

# Default parameters for RedCellProcessor
DEFAULT_PARAMETERS = dict(
    um_per_pixel=None,
    surround_iterations=2,
    fill_value=0.0,
    centered_width=40,
    centroid_method="median",
    window_kernel=np.hanning,
    phase_corr_eps=1e6,
    lowcut=12,
    highcut=250,
    order=3,
)

# Mapping of parameters to cache entries that are affected by the change
PARAM_CACHE_MAPPING = dict(
    surround_iterations=[],
    fill_value=[
        "centered_masks",
        "centered_reference",
        "filtered_centered_reference",
        "centered_reference_functional",
        "filtered_centered_reference_functional",
    ],
    centered_width=[
        "centered_masks",
        "centered_reference",
        "filtered_centered_reference",
        "centered_reference_functional",
        "filtered_centered_reference_functional",
    ],
    centroid_method=[
        "centroids",
        "centered_masks",
        "centered_reference",
        "filtered_centered_reference",
        "centered_reference_functional",
        "filtered_centered_reference_functional",
    ],
    window_kernel=[],
    phase_corr_eps=[],
    lowcut=["filtered_centered_reference", "filtered_centered_reference_functional"],
    highcut=["filtered_centered_reference", "filtered_centered_reference_functional"],
    order=["filtered_centered_reference", "filtered_centered_reference_functional"],
)


class RoiProcessor:
    """
    Process and analyze mask & fluorescence data across multiple image planes.

    This class handles the processing of mask & fluorescence data by managing masks
    and reference images across multiple planes, providing functionality for feature
    calculation and analysis.

    Attributes
    ----------
    root_dir : Path
        Path to the root directory where the data is stored.
    zpix : Union[np.ndarray, List[np.ndarray]]
        Numpy array containing the plane index for each ROI.
        Or, if volumetric=True, a list of numpy arrays containing the z-pixel indices for each ROI.
    ypix : List[np.ndarray]
        List of numpy arrays containing the y-pixel indices for each ROI.
    xpix : List[np.ndarray]
        List of numpy arrays containing the x-pixel indices for each ROI.
    lam : List[np.ndarray]
        List of numpy arrays containing the pixen intensities for each ROI.
    reference : np.ndarray
        3D numpy array containing reference images for each plane.
    functional_reference : Union[np.ndarray, None]
        3D numpy array containing functional reference images for each plane (optional).
    lx, ly, lz : int
        Dimensions of the imaging volume.
    num_rois : int
        Total number of ROIs across all planes.
    features : dict
        Computed features for all ROIs.
    feature_pipeline_methods : dict
        Mapping of feature pipeline names to their corresponding methods.
    feature_pipeline_dependencies : dict
        Mapping of feature pipeline names to dependencies on attributes of roi_processor instances.
    parameters : dict
        Dictionary containing all the preprocessing parameters used.
    _cache : dict
        Dictionary containing cached values of attributes that are expensive to compute.
    """

    def __init__(
        self,
        root_dir: Union[Path, str],
        zpix: Union[np.ndarray, List[np.ndarray]],
        ypix: List[np.ndarray],
        xpix: List[np.ndarray],
        lam: List[np.ndarray],
        reference: np.ndarray,
        functional_reference: Union[np.ndarray, None] = None,
        extra_features: Optional[Dict[str, List[np.ndarray]]] = None,
        volumetric: bool = False,
        autocompute: bool = True,
        use_saved: bool = True,
        save_features: bool = True,
        **kwargs: dict,
    ):
        """Initialize the RoiProcessor with ROI stats and reference images.

        Parameters
        ----------
        root_dir : Union[Path, str]
            Path to the root directory where the data is stored. This is used to save and load
            features from disk.
        zpix: Union[np.ndarray, List[np.ndarray]]
            Numpy array containing the plane index for each ROI.
            Or, if volumetric=True, a list of numpy arrays containing the z-pixel indices for each ROI.
        ypix: List[np.ndarray]
            List of numpy arrays containing the y-pixel indices for each ROI.
        xpix: List[np.ndarray]
            List of numpy arrays containing the x-pixel indices for each ROI.
        lam: List[np.ndarray]
            List of numpy arrays containing the pixen intensities for each ROI.
        reference : np.ndarray
            3D numpy array containing reference images for each plane. The first dimension should
            be the number of planes (lz), and the second and third dimensions should be the height
            and width of the reference images (notated as ly and lx).
        functional_reference : Union[np.ndarray, None], optional
            3D numpy array containing functional reference images for each plane (optional).
        extra_features : Optional[Dict[str, List[np.ndarray]]], optional
            Dictionary containing extra features to be added to each plane. Each key is the
            name of the feature and the value is a list of 1d numpy arrays with length equal
            to the number of ROIs in each plane. Default is None.
        volumetric : bool, optional
            Set as True to indicate that the data is volumetric (e.g. processed with suite3d). This
            will convert all processing operations to use volumetric information about ROI masks.
        autocompute : bool, optional
            If True, will automatically compute all standard features upon initialization. The only
            reason not to have this set to True is if you want the object for some other purpose or
            if you want to compute a subset of the features, which you can do manually. Default is True.
        use_saved : bool, optional
            If True, will attempt to load saved features from disk if they exist. Default is True.
        save_features : bool, optional
            If True, will save the computed features to disk. Default is True.
        **kwargs : dict
            Additional parameters to update the default parameters used for preprocessing.
        """
        if len(lam) != len(ypix) != len(xpix) != len(zpix):
            raise ValueError(
                "Lengths of mask data do not match each other (inspect lam, xpix, ypix, and zpix)!"
            )
        if not isinstance(reference, np.ndarray) or reference.ndim != 3:
            raise TypeError("reference must be a 3D numpy array")
        if (
            functional_reference is not None
            and functional_reference.shape != reference.shape
        ):
            raise TypeError(
                "Functional reference must have the same shape as reference"
            )

        root_dir = Path(root_dir)
        if not root_dir.is_dir():
            raise ValueError("root_dir must be existing directory.")

        # Initialize attributes
        self.zpix = zpix
        self.ypix = ypix
        self.xpix = xpix
        self.lam = lam
        self.lz, self.ly, self.lx = reference.shape
        self.num_rois = len(lam)
        self.reference = reference
        self.functional_reference = functional_reference
        self.volumetric = volumetric
        self.root_dir = root_dir
        self.save_features = save_features

        # Validate mask data for each ROI
        if self.volumetric:
            for lm, xp, yp, zp in zip(self.lam, self.xpix, self.ypix, self.zpix):
                if not (len(lm) == len(xp) == len(yp) == len(zp)):
                    raise ValueError("Mismatched lengths of mask data")
            if (
                max(max(x) for x in self.xpix) >= self.lx
                or max(max(y) for y in self.ypix) >= self.ly
                or max(max(z) for z in self.zpix) >= self.lz
            ):
                raise ValueError("Pixel indices exceed image dimensions")
        else:
            for lm, xp, yp in zip(self.lam, self.xpix, self.ypix):
                if not (len(lm) == len(xp) == len(yp)):
                    raise ValueError("Mismatched lengths of mask data")
            if (
                max(max(x) for x in self.xpix) >= self.lx
                or max(max(y) for y in self.ypix) >= self.ly
            ):
                raise ValueError("Pixel indices exceed image dimensions")

        # Store flattened mask data for some optimized implementations
        self._flattened_roi_data = utils.flatten_roi_data(
            self.lam,
            self.ypix,
            self.xpix,
            self.zpix if self.volumetric else None,
        )

        # Initialize feature and pipeline dictionary
        self.features = {}
        self.feature_pipeline_methods = {}
        self.feature_pipeline_dependencies = {}

        # Initialize preprocessing cache
        self._cache = {}

        # If extra features are provided, validate and store
        if extra_features is not None:
            if not isinstance(extra_features, dict):
                raise TypeError("Extra features must be a dictionary")
            for name, value in extra_features.items():
                if not isinstance(name, str):
                    raise TypeError("Extra feature values must be a numpy array")
                if not isinstance(value, np.ndarray) or not len(value) == self.num_rois:
                    raise TypeError(
                        "Extra feature values must be a numpy array with length equal to the number of ROIs"
                    )
                self.add_feature(name, value)

        # Establish preprocessing parameters
        self.parameters = deepcopy(DEFAULT_PARAMETERS)
        if set(kwargs) - set(DEFAULT_PARAMETERS):
            raise ValueError(
                f"Invalid parameter(s): {', '.join(set(kwargs) - set(DEFAULT_PARAMETERS))}"
            )
        self.parameters.update(kwargs)

        # register feature pipelines
        for pipeline in standard_pipelines:
            self.register_feature_pipeline(pipeline)

        if self.functional_reference is not None:
            for pipeline in functional_pipelines:
                self.register_feature_pipeline(pipeline)

        # Measure features
        if autocompute:
            self.compute_features(use_saved)

    def compute_features(self, use_saved: bool = True):
        """Compute all registered features for each ROI.

        FeaturePipelines are registered with the RoiProcessor instance, and each pipeline
        defines a method that computes a feature based on the attributes of the RoiProcessor
        instance. compute_features iterates over each pipeline and computes the feature values
        for each ROI. Resulting feature values are stored in the self.features dictionary.

        Parameters
        ----------
        use_saved : bool, optional
            If True, will attempt to load saved features from disk if they exist. Default is True.
        """
        from .io.base import load_feature, is_feature_saved

        for name, method in self.feature_pipeline_methods.items():
            if use_saved:
                if is_feature_saved(self.root_dir, name):
                    value = load_feature(self.root_dir, name)
                    if len(value) == self.num_rois:
                        self.add_feature(name, value)
                        # Skip recomputing the feature and move to next one
                        continue

            # If the feature is not saved or the shapes don't match, compute the feature again and add it
            self.add_feature(name, method(self))

    def add_feature(self, name: str, values: np.ndarray):
        """Add (or update) the name and values to the self.features dictionary.

        Parameters
        ----------
        name : str
            Name of the feature.
        values : np.ndarray
            Feature values for each ROI. Must have the same length as the number of ROIs across all planes.
        """
        from .io.base import save_feature

        if len(values) != self.num_rois:
            raise ValueError(
                f"Length of feature values ({len(values)}) for feature {name} must match number of ROIs ({self.num_rois})"
            )
        self.features[name] = values  # cache the feature values
        if self.save_features:
            # save to disk if requested
            save_feature(self.root_dir, name, values)

    def register_feature_pipeline(self, pipeline: FeaturePipeline):
        """Register a feature pipeline with the RoiProcessor instance.

        pipeline is a FeaturePipeline object that defines a method to compute a feature
        based on the attributes of the RoiProcessor instance. The method should take the
        RoiProcessor instance as an argument and return a numpy array of feature values.
        The dependencies attribute of the pipeline object should be a list of strings
        indicating the attributes of the RoiProcessor instance that the method depends on.
        If any of these attributes are updated, the feature will be recomputed.

        Parameters
        ----------
        pipeline : FeaturePipeline
            FeaturePipeline object that defines a method to compute a feature based on the
            attributes of the RoiProcessor instance.
        """
        if not isinstance(pipeline, FeaturePipeline):
            raise TypeError("Pipeline must be an instance of FeaturePipeline")
        if (
            pipeline.name in self.feature_pipeline_methods
            or pipeline.name in self.feature_pipeline_dependencies
        ):
            raise ValueError(
                f"A pipeline called {pipeline.name} has already been registered."
            )
        if not all(dep in self.parameters for dep in pipeline.dependencies):
            raise ValueError(
                f"The following dependencies for pipeline {pipeline.name} not found in parameters ({', '.join(pipeline.dependencies)})"
            )
        self.feature_pipeline_methods[pipeline.name] = pipeline.method
        self.feature_pipeline_dependencies[pipeline.name] = pipeline.dependencies

    def update_parameters(self, **kwargs: dict):
        """Update preprocessing parameters and clear affected cache entries.

        Preprocessing parameters are used to compute properties of self that are cached
        upon first access, and also for feature computation. When parameters are updated,
        the cache entries that are affected by the change are cleared so they can be
        recomputed with the new parameters when accessed again. Features are automatically
        regenerated if they depend on the updated parameters and have already been computed.

        Parameter dependencies are indicated in the PARAM_CACHE_MAPPING dictionary.
        Feature dependencies are indicated in the feature_pipeline_dependencies dictionary.

        Parameters
        ----------
        **kwargs : dict
            New values to update in the initial dictionary. Must be a subset of the keys in
            initial, otherwise a ValueError will be raised.

        Returns
        -------
        dict
            Updated dictionary of parameters.
        """
        # First check if any invalid parameters are provided
        extra_kwargs = set(kwargs) - set(self.parameters)
        if extra_kwargs:
            raise ValueError(f"Invalid parameter(s): {', '.join(extra_kwargs)}")

        # For every changed parameter, identify affected cache / features
        affected_cache = []
        affected_features = []
        for key, value in kwargs.items():
            if key in self.parameters and self.parameters[key] != value:
                affected_cache.extend(PARAM_CACHE_MAPPING.get(key, []))
                for (
                    pipeline,
                    dependencies,
                ) in self.feature_pipeline_dependencies.items():
                    if key in dependencies:
                        affected_features.append(pipeline)
                self.parameters[key] = value

        # Clear affected cache to be recomputed lazily whenever it is needed again
        for cache_key in set(affected_cache):
            self._cache.pop(cache_key, None)

        # Recompute affected features if they have already been computed
        for feature_key in set(affected_features):
            if feature_key in self.features:
                self.add_feature(
                    feature_key, self.feature_pipeline_methods[feature_key](self)
                )

    def copy_with_params(self, params: dict):
        """Create a new processor instance with updated parameters.

        Parameters
        ----------
        params : dict
            New parameter values to update in the new instance. Must be a subset of the
            keys in DEFAULT_PARAMETERS, otherwise a ValueError will be raised.

        Returns
        -------
        RoiProcessor
            New instance of RoiProcessor with updated parameters.
        """
        copy_of_self = deepcopy(self)
        copy_of_self.update_parameters(**params)
        return copy_of_self

    @property
    def centroids(self):
        """Return the centroids of the ROIs in each plane.

        Centroids are two lists of the y-centroid and x-centroid for each ROI,
        concatenated across planes. The centroid method is determined by the
        centroid_method attribute. Centroids are always returned as integers.

        Returns
        -------
        Tuple[np.ndarray]
            Tuple of two numpy arrays, the y-centroids and x-centroids.
        """
        if "centroids" not in self._cache:
            zc, yc, xc = utils.get_roi_centroids(
                self.lam,
                self.zpix,
                self.ypix,
                self.xpix,
                method=self.parameters["centroid_method"],
                asint=True,
            )
            centroids = dict(zc=zc, yc=yc, xc=xc)
            self._cache["centroids"] = centroids
        return self._cache["centroids"]

    @property
    def centered_masks(self):
        """Return the centered mask images for each ROI.

        Returns
        -------
        np.ndarray
            The centered mask images of each ROI.
            If volumetric=False, has shape (numROIs, centered_width*2+1, centered_width*2+1).
            If volumetric=True, has shape (numROIs, num_planes, centered_width*2+1, centered_width*2+1).
        """
        if "centered_masks" not in self._cache:
            centered_masks = utils.get_centered_masks(
                self._flattened_roi_data,
                self.centroids,
                width=self.parameters["centered_width"],
                fill_value=self.parameters["fill_value"],
                num_planes=self.lz,
                volumetric=self.volumetric,
            )
            self._cache["centered_masks"] = centered_masks
        return self._cache["centered_masks"]

    @property
    def centered_reference(self):
        """Return the centered reference image for each ROI.

        Returns
        -------
        np.ndarray
            The centered reference image around each ROI.
            If volumetric=False, has shape (numROIs, centered_width*2+1, centered_width*2+1).
            If volumetric=True, has shape (numROIs, num_planes, centered_width*2+1, centered_width*2+1).
        """
        if "centered_reference" not in self._cache:
            centered_reference = utils.get_centered_reference(
                self.reference,
                self.centroids,
                width=self.parameters["centered_width"],
                fill_value=self.parameters["fill_value"],
                volumetric=self.volumetric,
            )
            self._cache["centered_reference"] = centered_reference
        return self._cache["centered_reference"]

    @property
    def filtered_reference(self):
        """Return the filtered reference image for each ROI.

        Uses a Butterworth bandpass filter to filter the reference image.

        Returns
        -------
        np.ndarray
            The filtered reference image for each ROI, with shape (numROIs, lx, ly)
        """
        if "filtered_reference" not in self._cache:
            bpf_parameters = dict(
                lowcut=self.parameters["lowcut"],
                highcut=self.parameters["highcut"],
                order=self.parameters["order"],
            )
            filtered_reference = filter(
                self.reference, "butterworth_bpf", **bpf_parameters
            )
            self._cache["filtered_reference"] = filtered_reference
        return self._cache["filtered_reference"]

    @property
    def filtered_centered_reference(self):
        """Return the filtered centered reference image for each ROI.

        Uses a Butterworth bandpass filter to filter the reference image, then generates
        a centered reference stack around each ROI using the filtered reference.

        Returns
        -------
        np.ndarray
            The filtered centered reference image around each ROI, with shape (numROIs, centered_width*2+1, centered_width*2+1)
        """
        if "filtered_centered_reference" not in self._cache:
            filtered_centered_reference = utils.get_centered_reference(
                self.filtered_reference,
                self.centroids,
                width=self.parameters["centered_width"],
                fill_value=self.parameters["fill_value"],
                volumetric=self.volumetric,
            )
            self._cache["filtered_centered_reference"] = filtered_centered_reference
        return self._cache["filtered_centered_reference"]

    @property
    def centered_reference_functional(self):
        """Return the centered reference image for each ROI based on the functional reference image.

        Returns
        -------
        np.ndarray
            The centered reference image around each ROI, with shape (numROIs, centered_width*2+1, centered_width*2+1)

        Raises
        ------
        ValueError
            If functional reference are not available.
        """
        if self.functional_reference is None:
            raise ValueError("Functional reference are not available")
        if "centered_reference_functional" not in self._cache:
            centered_reference = utils.get_centered_reference(
                self.functional_reference,
                self.centroids,
                width=self.parameters["centered_width"],
                fill_value=self.parameters["fill_value"],
                volumetric=self.volumetric,
            )
            self._cache["centered_reference_functional"] = centered_reference
        return self._cache["centered_reference_functional"]

    @property
    def filtered_reference_functional(self):
        """Return the filtered reference image for each ROI based on the functional reference image.

        Uses a Butterworth bandpass filter to filter the reference image.

        Returns
        -------
        np.ndarray
            The filtered reference image for each ROI, with shape (numROIs, lx, ly)

        Raises
        ------
        ValueError
            If functional reference are not available.
        """
        if self.functional_reference is None:
            raise ValueError("Functional reference are not available")
        if "filtered_reference_functional" not in self._cache:
            bpf_parameters = dict(
                lowcut=self.parameters["lowcut"],
                highcut=self.parameters["highcut"],
                order=self.parameters["order"],
            )
            filtered_reference = filter(
                self.functional_reference,
                "butterworth_bpf",
                **bpf_parameters,
            )
            self._cache["filtered_reference_functional"] = filtered_reference
        return self._cache["filtered_reference_functional"]

    @property
    def filtered_centered_reference_functional(self):
        """Return the filtered centered reference image for each ROI based on the functional reference image.

        Uses a Butterworth bandpass filter to filter the reference image, then generates
        a centered reference stack around each ROI using the filtered reference.

        Returns
        -------
        np.ndarray
            The filtered centered reference image around each ROI, with shape (numROIs, centered_width*2+1, centered_width*2+1)

        Raises
        ------
        ValueError
            If functional reference are not available.
        """
        if self.functional_reference is None:
            raise ValueError("Functional reference are not available")
        if "filtered_centered_reference_functional" not in self._cache:
            filtered_centered_reference = utils.get_centered_reference(
                self.filtered_reference_functional,
                self.centroids,
                width=self.parameters["centered_width"],
                fill_value=self.parameters["fill_value"],
                volumetric=self.volumetric,
            )
            self._cache["filtered_centered_reference_functional"] = (
                filtered_centered_reference
            )
        return self._cache["filtered_centered_reference_functional"]
