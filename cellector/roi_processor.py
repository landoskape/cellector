from typing import List, Optional, Dict, Union
from pathlib import Path
from copy import deepcopy
import numpy as np
from . import io
from . import utils
from .filters import filter
from .feature_pipelines import FeaturePipeline, standard_pipelines

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
    fill_value=["centered_masks", "centered_references", "filtered_centered_references"],
    centered_width=["centered_masks", "centered_references", "filtered_centered_references"],
    centroid_method=["yc", "xc", "centered_masks", "centered_references", "filtered_centered_references"],
    window_kernel=[],
    phase_corr_eps=[],
    lowcut=["filtered_centered_references"],
    highcut=["filtered_centered_references"],
    order=["filtered_centered_references"],
)


class RoiProcessor:
    """
    Process and analyze mask & fluorescence data across multiple image planes.

    This class handles the processing of mask & fluorescence data by managing masks
    and reference images across multiple planes, providing functionality for feature
    calculation and analysis.

    # TODO: Add attributes and methods to the class docstring.
    """

    def __init__(
        self,
        stats: List[np.ndarray],
        references: List[np.ndarray],
        root_dir: Union[Path, str],
        extra_features: Optional[Dict[str, List[np.ndarray]]] = None,
        autocompute: bool = True,
        use_saved: bool = True,
        **kwargs: dict,
    ):
        """Initialize the RoiProcessor with ROI stats and reference images.

        Parameters
        ----------
        stats: List[np.array[dict]]
            List of dictionaries containing ROI statistics for each plane.
            required keys: 'lam', 'xpix', 'ypix', which are lists of numbers corresponding to
            the weight of each pixel, and the x and y indices of each pixel in the mask
        references : List[np.ndarray]
            List of reference images for each plane. Each element is a 2D
            numpy array with shape (lx, ly).
        root_dir : Union[Path, str]
            Path to the root directory where the data is stored. This is used to save and load
            features from disk.
        extra_features : Dict[str, np.ndarray], optional
            Dictionary containing extra features to be added to each plane. Each key is the
            name of the feature and the value is a list of 1d numpy arrays with length equal
            to the number of ROIs in each plane. Default is None.
        autocompute : bool, optional
            If True, will automatically compute all standard features upon initialization. The only
            reason not to have this set to True is if you want the object for some other purpose or
            if you want to compute a subset of the features, which you can do manually. Default is True.
        use_saved : bool, optional
            If True, will attempt to load saved features from disk if they exist. Default is True.
        **kwargs : dict
            Additional parameters to update the default parameters used for preprocessing.
        """
        # Validate inputs are lists
        if not isinstance(stats, list) or not isinstance(references, list):
            raise TypeError("Stats and references must be lists.")

        if not stats or not references:
            raise ValueError("Stats and references lists cannot be empty")

        # Validate number of planes match
        if len(stats) != len(references):
            raise ValueError(f"Number of mask arrays ({len(stats)}) must match reference images ({len(references)})")

        if not all(isinstance(ref, np.ndarray) for ref in references) or not all(ref.ndim == 2 for ref in references):
            raise ValueError("All reference images must be 2D numpy arrays")

        root_dir = Path(root_dir)
        if not root_dir.is_dir():
            raise ValueError("Root directory must be a valid directory path")

        # Initialize attributes
        self.num_planes = len(stats)
        self.lx, self.ly = references[0].shape
        self.rois_per_plane = [len(stat) for stat in stats]
        self.num_rois = sum(self.rois_per_plane)
        self.references = references
        self.root_dir = root_dir

        # Extract mask data from stats dictionaries
        lam, ypix, xpix = utils.transpose([utils.get_roi_data(stat) for stat in stats])

        # Validate mask data for each plane
        for iplane, (lm, xp, yp) in enumerate(zip(lam, xpix, ypix)):
            max_xp = max(max(x) for x in xp)
            max_yp = max(max(y) for y in yp)
            if max_xp >= self.lx or max_yp >= self.ly:
                raise ValueError(f"Pixel indices exceed image dimensions in plane {iplane}")
            for l, x, y in zip(lm, xp, yp):
                if not (len(l) == len(x) == len(y)):
                    raise ValueError(f"Mismatched lengths of mask data in plane {iplane}")

        # Store mask data as concatenated planes
        self.lam = utils.cat_planes(lam)
        self.ypix = utils.cat_planes(ypix)
        self.xpix = utils.cat_planes(xpix)

        # Get plane index to each ROI
        self.plane_idx = np.repeat(np.arange(self.num_planes), self.rois_per_plane)

        # Store flattened mask data for some optimized implementations
        lam_flat, ypix_flat, xpix_flat, flat_roi_idx = utils.flatten_roi_data(self.lam, self.ypix, self.xpix)
        self.lam_flat = lam_flat
        self.ypix_flat = ypix_flat
        self.xpix_flat = xpix_flat
        self.flat_roi_idx = flat_roi_idx

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
            for name, values in extra_features.items():
                if not isinstance(name, str):
                    raise TypeError("Extra feature values must be a numpy array")
                if not isinstance(values, list) and not all(isinstance(v, np.ndarray) for v in values):
                    raise TypeError("Extra feature values must be a list of numpy arrays")
                if not all(v.ndim == 1 for v in values) or not all(len(v) == nroi for v, nroi in zip(values, self.rois_per_plane)):
                    raise ValueError("Extra feature values must be 1D numpy arrays with length equal to the number of ROIs for each plane.")
                self.add_feature(name, utils.cat_planes(values))

        # Establish preprocessing parameters
        self.parameters = deepcopy(DEFAULT_PARAMETERS)
        if set(kwargs) - set(DEFAULT_PARAMETERS):
            raise ValueError(f"Invalid parameter(s): {', '.join(set(kwargs) - set(DEFAULT_PARAMETERS))}")
        self.parameters.update(kwargs)

        # register feature pipelines
        for pipeline in standard_pipelines:
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
        for name, method in self.feature_pipeline_methods.items():
            if use_saved:
                if io.is_feature_saved(self.root_dir, name):
                    value = io.load_saved_feature(self.root_dir, name)
                    if len(value) == self.num_rois:
                        self.add_feature(name, value)
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
        if len(values) != self.num_rois:
            raise ValueError(f"Length of feature values ({len(values)}) for feature {name} must match number of ROIs ({self.num_rois})")
        self.features[name] = values

    def register_feature_pipeline(self, pipeline: FeaturePipeline):
        """Register a feature pipeline with the RoiProcessor instance."""
        if not isinstance(pipeline, FeaturePipeline):
            raise TypeError("Pipeline must be an instance of FeaturePipeline")
        if pipeline.name in self.feature_pipeline_methods or pipeline.name in self.feature_pipeline_dependencies:
            raise ValueError(f"A pipeline called {pipeline.name} has already been registered.")
        if not all(dep in self.parameters for dep in pipeline.dependencies):
            raise ValueError(f"The following dependencies for pipeline {pipeline.name} not found in parameters ({', '.join(pipeline.dependencies)})")
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
                for pipeline, dependencies in self.feature_pipeline_dependencies.items():
                    if key in dependencies:
                        affected_features.append(pipeline)
                self.parameters[key] = value

        # Clear affected cache to be recomputed lazily whenever it is needed again
        for cache_key in set(affected_cache):
            self._cache.pop(cache_key, None)

        # Recompute affected features if they have already been computed
        for feature_key in set(affected_features):
            if feature_key in self.features:
                self.add_feature(feature_key, self.feature_pipeline_methods[feature_key](self))

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
        self_copy = deepcopy(self)
        self_copy.update_parameters(**params)
        return self_copy

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
        if "yc" not in self._cache or "xc" not in self._cache:
            yc, xc = utils.get_roi_centroids(
                self.lam,
                self.ypix,
                self.xpix,
                method=self.parameters["centroid_method"],
                asint=True,
            )
            self._cache["yc"] = yc
            self._cache["xc"] = xc
        return self._cache["yc"], self._cache["xc"]

    @property
    def yc(self):
        """Return the y-centroids of the ROIs in each plane.

        Returns
        -------
        np.ndarray
            The y-centroids of the ROIs.
        """
        return self.centroids[0]

    @property
    def xc(self):
        """Return the x-centroids of the ROIs in each plane.

        Returns
        -------
        np.ndarray
            The x-centroids of the ROIs.
        """
        return self.centroids[1]

    @property
    def centered_masks(self):
        """Return the centered mask images for each ROI.

        Returns
        -------
        np.ndarray
            The centered mask images of each ROI, with shape (numROIs, centered_width*2+1, centered_width*2+1)
        """
        if "centered_masks" not in self._cache:
            centered_masks = utils.get_centered_masks(
                self.lam_flat,
                self.ypix_flat,
                self.xpix_flat,
                self.flat_roi_idx,
                self.centroids,
                width=self.parameters["centered_width"],
                fill_value=self.parameters["fill_value"],
            )
            self._cache["centered_masks"] = centered_masks
        return self._cache["centered_masks"]

    @property
    def centered_references(self):
        """Return the centered references image for each ROI.

        Returns
        -------
        np.ndarray
            The centered references image around each ROI, with shape (numROIs, centered_width*2+1, centered_width*2+1)
        """
        if "centered_references" not in self._cache:
            centered_references = utils.get_centered_references(
                self.references,
                self.plane_idx,
                self.centroids,
                width=self.parameters["centered_width"],
                fill_value=self.parameters["fill_value"],
            )
            self._cache["centered_references"] = centered_references
        return self._cache["centered_references"]

    @property
    def filtered_references(self):
        """Return the filtered reference image for each ROI.

        Uses a Butterworth bandpass filter to filter the reference image.

        Returns
        -------
        np.ndarray
            The filtered reference image for each ROI, with shape (numROIs, lx, ly)
        """
        if "filtered_references" not in self._cache:
            bpf_parameters = dict(
                lowcut=self.parameters["lowcut"],
                highcut=self.parameters["highcut"],
                order=self.parameters["order"],
            )
            filtered_references = filter(np.stack(self.references), "butterworth_bpf", **bpf_parameters)
            self._cache["filtered_references"] = filtered_references
        return self._cache["filtered_references"]

    @property
    def filtered_centered_references(self):
        """Return the filtered centered references image for each ROI.

        Uses a Butterworth bandpass filter to filter the reference image, then generates
        a centered reference stack around each ROI using the filtered reference.

        Returns
        -------
        np.ndarray
            The filtered centered references image around each ROI, with shape (numROIs, centered_width*2+1, centered_width*2+1)
        """
        if "filtered_centered_references" not in self._cache:
            filtered_centered_references = utils.get_centered_references(
                self.filtered_references,
                self.plane_idx,
                self.centroids,
                width=self.parameters["centered_width"],
                fill_value=self.parameters["fill_value"],
            )
            self._cache["filtered_centered_references"] = filtered_centered_references
        return self._cache["filtered_centered_references"]

    @property
    def mask_volume(self):
        """Return the mask volume for each ROI.

        The output is a 3D array where each slice represents the mask data for each ROI,
        with zeros outside the footprint of the ROI.

        Returns
        -------
        np.ndarray
            The mask volume for each ROI, with shape (numROIs, ly, lx)
        """
        if "mask_volume" not in self._cache:
            mask_volume = utils.get_mask_volume(
                self.lam_flat,
                self.ypix_flat,
                self.xpix_flat,
                self.flat_roi_idx,
                self.num_rois,
                (self.ly, self.lx),
            )
            self._cache["mask_volume"] = mask_volume
        return self._cache["mask_volume"]


# TODO: Implement the equivalent of this function (probably compare other RedCell objects to one reference point)
# def compareFeatureCutoffs(*vrexp, roundValue=None):
#     features = [
#         "parametersRedS2P.minMaxCutoff",
#         "parametersRedDotProduct.minMaxCutoff",
#         "parametersRedPearson.minMaxCutoff",
#         "parametersRedPhaseCorrelation.minMaxCutoff",
#     ]
#     dfDict = {"session": [ses.sessionPrint() for ses in vrexp]}

#     def getFeatName(name):
#         cname = name[name.find("Red") + 3 : name.find(".")]
#         return cname  # cname+'_min', cname+'_max'

#     for feat in features:
#         dfDict[getFeatName(feat)] = [None] * len(vrexp)

#     for idx, ses in enumerate(vrexp):
#         for feat in features:
#             cdata = ses.loadone(feat)
#             if cdata.dtype == object and cdata.item() is None:
#                 cdata = [None, None]
#             else:
#                 if roundValue is not None:
#                     cdata = np.round(cdata, roundValue)
#             dfDict[getFeatName(feat)][idx] = cdata

#     print(pd.DataFrame(dfDict))
#     return None
