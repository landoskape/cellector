from typing import List, Optional
from copy import deepcopy
import numpy as np
from . import utils
from . import features
from .filters import filter

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

# Mapping of parameters to features that are affected by the change
PARAM_FEATURE_MAPPING = dict(
    surround_iterations=["in_vs_out", "corr_coef"],
    centered_width=["phase_corr", "in_vs_out", "corr_coef"],
    centroid_method=["phase_corr", "in_vs_out", "corr_coef"],
    window_kernel=["phase_corr"],
    phase_corr_eps=["phase_corr"],
    lowcut=["dot_product", "corr_coef"],
    highcut=["dot_product", "corr_coef"],
    order=["dot_product", "corr_coef"],
)


class RedCellProcessor:
    """
    Process and analyze red cell data across multiple image planes.

    This class handles the processing of red cell data by managing masks and reference
    images across multiple planes, providing functionality for feature calculation
    and analysis.

    Attributes
    ----------
    feature_names : List[str]
        Standard features computed for each cell.
    feature_values : List[np.ndarray]
        Values of the standard features for each cell.
    num_planes : int
        Number of planes in the dataset.
    lx : int
        X dimension of the masks/images.
    ly : int
        Y dimension of the masks/images.
    num_rois : List[int]
        Number of ROIs in each plane.
    masks : List[np.ndarray]
        List of masks for each plane, shape (numROIs, lx, ly) per plane.
    reference : List[np.ndarray]
        List of reference images, shape (lx, ly) per plane.
    um_per_pixel : float
        Conversion factor from pixels to micrometers.
    """

    def __init__(self, stats: List[np.ndarray], references: List[np.ndarray], red_s2p: Optional[List[np.ndarray]] = None, **kwargs: dict):
        """Initialize the RedCellProcessor with ROI stats and reference images.

        Parameters
        ----------
        stats: List[np.array[dict]]
            List of dictionaries containing ROI statistics for each plane.
            required keys: 'lam', 'xpix', 'ypix', which are lists of numbers corresponding to
            the weight of each pixel, and the x and y indices of each pixel in the mask
        references : List[np.ndarray]
            List of reference images for each plane. Each element is a 2D
            numpy array with shape (lx, ly).
        red_s2p : List[np.ndarray], optional
            List of red cell probability from suite2p (or anywhere else) for each plane.
            Each element is a 1d numpy array with length equal to the number of ROIs in that plane.
        **kwargs : dict
            Additional parameters to update the default parameters used for preprocessing.
        """
        # Validate inputs are lists
        if not isinstance(stats, list) or not isinstance(references, list):
            raise TypeError("Both masks and reference must be lists of numpy arrays")

        if not stats or not references:  # Check for empty lists
            raise ValueError("Masks and reference lists cannot be empty")

        # Validate number of planes match
        if len(stats) != len(references):
            raise ValueError(f"Number of mask arrays ({len(stats)}) must match number of reference images ({len(references)})")

        if not all(ref.ndim == 2 for ref in references):
            raise ValueError("All reference images must be 2D")

        # Initialize attributes
        self.num_planes = len(stats)
        self.lx, self.ly = references[0].shape
        self.rois_per_plane = [len(stat) for stat in stats]
        self.num_rois = sum(self.rois_per_plane)
        self.references = references

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

        # Initialize feature dictionary
        self.features = {}

        # Initialize preprocessing cache
        self._cache = {}

        # If red_s2p is provided, validate and store
        if red_s2p is not None:
            if len(red_s2p) != self.num_planes:
                raise ValueError(f"Number of red_s2p arrays ({len(red_s2p)}) must match number of planes ({self.num_planes})")
            if not all(len(s2p) == nroi for s2p, nroi in zip(red_s2p, self.rois_per_plane)):
                raise ValueError("Length of red_s2p arrays must match number of ROIs in each plane")
            self.add_feature("red_s2p", utils.cat_planes(red_s2p))

        # Establish preprocessing parameters
        self.parameters = deepcopy(DEFAULT_PARAMETERS)
        if set(kwargs) - set(DEFAULT_PARAMETERS):
            raise ValueError(f"Invalid parameter(s): {', '.join(set(kwargs) - set(DEFAULT_PARAMETERS))}")
        self.parameters.update(kwargs)

    def update_parameters(self, **kwargs: dict):
        """Update preprocessing parameters and clear affected cache entries.

        Preprocessing parameters are used to compute properties of self that are cached
        upon first access, and also for feature computation. When parameters are updated,
        the cache entries that are affected by the change are cleared so they can be
        recomputed with the new parameters when accessed again. Features are automatically
        regenerated if they depend on the updated parameters.

        For a list of parameters that affect each cache entry or feature, see the two
        global dictionaries called PARAM_CACHE_MAPPING and PARAM_FEATURE_MAPPING.

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
        extra_kwargs = set(kwargs) - set(self.parameters)
        if extra_kwargs:
            raise ValueError(f"Invalid parameter(s): {', '.join(extra_kwargs)}")
        affected_cache = []
        affected_features = []
        for key, value in kwargs.items():
            if key in self.parameters and self.parameters[key] != value:
                affected_cache.extend(PARAM_CACHE_MAPPING.get(key, []))
                affected_features.extend(PARAM_FEATURE_MAPPING.get(key, []))
                self.parameters[key] = value
        for cache_key in set(affected_cache):
            self._cache.pop(cache_key, None)
        for feature_key in set(affected_features):
            self._feature_method(feature_key)()

    def copy_with_params(self, params: dict):
        """Create a new processor instance with updated parameters.

        Parameters
        ----------
        params : dict
            New parameter values to update in the new instance. Must be a subset of the
            keys in DEFAULT_PARAMETERS, otherwise a ValueError will be raised.

        Returns
        -------
        RedCellProcessor
            New instance of RedCellProcessor with updated parameters.
        """
        self_copy = deepcopy(self)
        self_copy.update_parameters(**params)
        return self_copy

    def _feature_method(self, feature_name: str):
        """Return the method to compute the given feature name."""
        feature_methods = dict(
            phase_corr=self.compute_phase_correlation,
            dot_product=self.compute_dot_product,
        )
        if feature_name not in feature_methods:
            raise ValueError(f"Feature {feature_name} not found in feature registry")
        return feature_methods[feature_name]

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

    def compute_phase_correlation(self, return_values: bool = False):
        """Compute the phase correlation between the masks and reference images.

        Parameters
        ----------
        return_values : bool, optional
            If True, will return the computed feature values. Default is False.

        Returns
        -------
        np.ndarray
            The phase correlation between the masks and reference images across planes.
            Default behavior is to not return the feature values, and instead to store them
            in the self.features dictionary of the RedCellProcessor instance.

        See Also
        --------
        features.phase_correlation_zero : Function that computes the phase correlation values.
        """
        # Input to phase correlation is centered masks and centered references
        centered_masks = self.centered_masks
        centered_references = self.centered_references

        # Window the centered masks and references
        windowed_masks = filter(centered_masks, "window", kernel=self.parameters["window_kernel"])
        windowed_references = filter(centered_references, "window", kernel=self.parameters["window_kernel"])

        # Phase correlation requires windowing
        phase_corr = features.phase_correlation_zero(windowed_masks, windowed_references, eps=self.parameters["phase_corr_eps"])
        self.add_feature("phase_corr", phase_corr)
        if return_values:
            return phase_corr

    def compute_dot_product(self, return_values: bool = False):
        """Compute the dot product between the masks and filtered reference images.

        Parameters
        ----------
        return_values : bool, optional
            If True, will return the computed feature values. Default is False.

        Returns
        -------
        np.ndarray
            The dot product between the masks and reference images across planes.
            Default behavior is to not return the feature values, and instead to store them
            in the self.features dictionary of the RedCellProcessor instance.

        See Also
        --------
        features.dot_product : Function that computes the dot product values.
        features.dot_product_array : Alternative that uses mask images instead of weights and pixel indices.
        """
        dot_product = features.dot_product(self.lam, self.ypix, self.xpix, self.plane_idx, self.filtered_references)
        self.add_feature("dot_product", dot_product)
        if return_values:
            return dot_product

    def compute_corr_coef(self, return_values: bool = False):
        """Compute the correlation coefficient between the masks and reference images.

        Parameters
        ----------
        return_values : bool, optional
            If True, will return the computed feature values. Default is False.

        Returns
        -------
        np.ndarray
            The correlation coefficient between the masks and reference images across planes.
            Default behavior is to not return the feature values, and instead to store them
            in the self.features dictionary of the RedCellProcessor instance.

        See Also
        --------
        features.correlation_coefficient : Function that computes the correlation coefficient values.
        """
        masks_surround, references_surround = utils.surround_filter(
            self.centered_masks,
            self.filtered_centered_references,
            iterations=self.parameters["surround_iterations"],
        )
        corr_coef = features.compute_correlation(masks_surround, references_surround)
        self.add_feature("corr_coef", corr_coef)
        if return_values:
            return corr_coef

    def compute_in_vs_out(self, return_values: bool = False):
        """Compute the in vs. out feature for each ROI.

        The in vs. out feature is the ratio of the dot product of the mask and reference
        image inside the mask to the dot product inside plus outside the mask.

        Parameters
        ----------
        return_values : bool, optional
            If True, will return the computed feature values. Default is False.

        Returns
        -------
        np.ndarray
            The in vs. out feature for each ROI. Default behavior is to not return the
            feature values, and instead to store them in the self.features dictionary of
            the RedCellProcessor instance.
        """
        in_vs_out = features.in_vs_out(
            self.centered_masks,
            self.centered_references,
            iterations=self.parameters["surround_iterations"],
        )
        self.add_feature("in_vs_out", in_vs_out)
        if return_values:
            return in_vs_out

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
