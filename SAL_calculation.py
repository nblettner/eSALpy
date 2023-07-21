import numpy as np
import xarray as xr
import scipy.ndimage as snd
from skimage.measure import regionprops, regionprops_table, label
import dask
from tqdm import tqdm


def trans_xr_np(x):
    """
    Transform to numpy arrays if input is a xarray dataarray.
    """
    try:
        x = x.values
    except:
        pass
    return x


def calc_dist(a, b):
    """
    Calculate the distance between two points or two sets of points.
    Spatial dimension must be last.
    """

    # function only for distances on 2D grid
    assert (a.shape[-1] == 2) and (b.shape[-1] == 2)

    d = ((a[..., 0] - b[..., 0]) ** 2 + (a[..., 1] - b[..., 1]) ** 2) ** 0.5

    return d


def calc_crps(a, b):
    """
    Calculate the continuous ranked probability score (CRPS).
    """

    a = np.atleast_1d(a)
    b = np.atleast_1d(b)

    boundaries = np.sort(np.concatenate((a, b)))

    widths = boundaries[1:] - boundaries[:-1]

    ind_a = 0
    ind_b = 0
    cdf_a = []
    cdf_b = []

    for i in range(len(widths)):
        if widths[i] != 0:
            if boundaries[i] in a:
                ind_a += 1
            if boundaries[i] in b:
                ind_b += 1

        cdf_a.append(ind_a / len(a))
        cdf_b.append(ind_b / len(b))

    integral = ((np.array(cdf_a) - np.array(cdf_b)) ** 2 * widths).sum()

    return integral


def calc_region_properties(fields, thld=0):
    """
    Calculate regional properties of fields dependent on threshold value.
    
    Parameters:
    fields: 3D array with spatial dimension at position 2 and 3, and optional
        ensemble dimension on position 1.
    thld: threshold to discriminate rainfall features from surrounding.
    
    Returns:
    dictionary of regional properties.
    """

    # mask used for properties of total domain
    mask = np.ones(fields.shape[1:]).astype(int)

    # initialize output variables (volume, total center of mass,
    # feature distances, number of features)
    V = []
    R_center = []
    fDist = []
    nF = []
    for member in fields:

        # total center of mass
        domain = regionprops(label_image=mask, intensity_image=member)
        Rf_center = np.array(domain[0].weighted_centroid)
        R_center.append(Rf_center)

        # ===================================
        # feature properties

        # identify features
        apply_thld = np.copy(member)
        apply_thld[apply_thld <= thld] = 0
        l, m = snd.label(apply_thld)
        features = regionprops(label_image=l, intensity_image=member)

        # local center of mass, area, mean, max
        Ro_area, Ro_mean, Ro_max, Ro_center = [], [], [], []
        # n_ = 0
        for f in features:
            Ro_area.append(f.area)
            Ro_mean.append(f.mean_intensity)
            Ro_max.append(f.max_intensity)
            Ro_center.append(f.weighted_centroid)
            # n_ += 1

        # sums of rain in features
        R_n = np.array(Ro_area) * np.array(Ro_mean)

        # ===================================
        # calculations that reduce feature dimension

        # volume normalized by domain sum of rain
        V.append((R_n**2 / np.array(Ro_max)).sum() / R_n.sum())

        # weightet distances of center of masses to
        # total center of mass normalized by domain sum of rain
        fDist.append((R_n * calc_dist(Rf_center, np.array(Ro_center))).sum() / R_n.sum())

        # number of features
        nF.append(len(features))

    regionProps = dict(
        V=np.array(V),
        R_center=np.array(R_center),
        fDist=np.array(fDist),
        nF=np.array(nF),
    )

    return regionProps


def calc_mSAL(a, b, rProp_a, rProp_b, maxDist):
    """
    Calculate SAL parameters of the individual ensemble members (mSAL).
    
    Parameters:
    a: reconstruction field / ensemble
    b: reference field / ensemble
    rProp_a: regional properties of a
    rProp_b: regional properties of b
    maxDist: maximum possible distance on the grid
    
    Returns:
    dictionary of mSAL parameters.
    """

    # structure
    S = (rProp_a["V"] - np.mean(rProp_b["V"])) / (
        0.5 * (rProp_a["V"] + np.mean(rProp_b["V"]))
    )

    # amplitude
    a_ = np.mean(a, axis=(1, 2))
    b_ = np.mean(b, axis=(1, 2))
    A = (a_ - np.mean(b_)) / (0.5 * (a_ + np.mean(b_)))

    # location
    L1 = calc_dist(rProp_a["R_center"], np.mean(rProp_b["R_center"], axis=0)) / maxDist
    L2 = 2 * np.abs(rProp_a["fDist"] - np.mean(rProp_b["fDist"])) / maxDist
    L = L1 + L2

    return dict(S=S, A=A, L=L, L1=L1)


def calc_eSAL(a, b, rProp_a, rProp_b, maxDist):
    """
    Calculate eSAL parameters of the whole ensemble.
    
    Parameters:
    a: reconstruction field / ensemble
    b: reference field / ensemble
    rProp_a: regional properties of a
    rProp_b: regional properties of b
    maxDist: maximum possible distance on the grid
    
    Returns:
    dictionary of (e)SAL parameters
    """

    # structure
    eS = (np.mean(rProp_a["V"]) - np.mean(rProp_b["V"])) / (
        0.5 * (np.mean(rProp_a["V"]) + np.mean(rProp_b["V"]))
    )

    # amplitude
    a_ = np.mean(a, axis=(1, 2))
    b_ = np.mean(b, axis=(1, 2))
    eA = (np.mean(a_) - np.mean(b_)) / (0.5 * (np.mean(a_) + np.mean(b_)))

    # location
    R_center_a = np.mean(rProp_a["R_center"], axis=0)
    R_center_b = np.mean(rProp_b["R_center"], axis=0)
    eL1 = calc_dist(R_center_a, R_center_b) / maxDist
    eL2 = 2 * calc_crps(rProp_a["fDist"] / maxDist, rProp_b["fDist"] / maxDist)
    eL = eL1 + eL2

    return dict(S=eS, A=eA, L=eL, L1=eL1)


def SAL_timestep(
    reconstruction,
    reference,
    time="xxx",
    thld_factor=1 / 15,
    quantile=0.95,
    fixed_thld=None,
    wet_thld=0,
    memberinfo=True,
    params=["S", "A", "L", "L1", "thld", "nF", "R_center"],
    as_dataset=True,
):

    """
    Wrapper for calculating (e)SAL parameters (and additional parameters) for one reconstruction (ensemble or single field) and reference (ensemble or single field). 
    
    Parameters:
    reconstruction, reference: Both can be a single field (shape = (ysize, xsize))
        or an ensemble (shape = (n_members, ysize, xsize)). In the latter case "SAL" is acutally "eSAL". Input fields can be numpy arrays or xarray dataarrays.
        
    time: index for the output (usually time but could be any string or number)
        
    thld_factor, quantile: used to define threshold contour, i.e. the minimum intensity required to belong to a feature.
        
    fixed_thld: give threshold directly without calculating via thld_factor and quantile
        
    wet_thld: value above which a pixel is considered wet
    
    memberinfo: In case reconstruction is an ensemble this returns parameters for the individual members also
        
    params: list of parameters that shall be returned
        
    as_dataset: if true returns an xarray dataset, otherwise a dictionary
        
    Returns:
        Xarray Dataset or dictionary of parameters
    """

    # ===================================
    # Pre-processing

    # if rec and ref are given as xarray dataarrays transform them to numpy arrays
    reconstruction = trans_xr_np(reconstruction)
    reference = trans_xr_np(reference)

    # assert right shape of input (either 2D: single field, or 3D: ensemble)
    assert (len(reconstruction.shape) in [2, 3]) and (
        len(reference.shape) in [2, 3]
    ), "Input must be 2- or 3-dimensional."

    # set flag whether SAL calculation can be conducted
    calculation_feasible = True

    # if single field and no ensemble is given change shape
    # (single fields are treated as ensemble of one member)
    if len(reconstruction.shape) == 2:
        reconstruction = reconstruction[np.newaxis, :]
    if len(reference.shape) == 2:
        reference = reference[np.newaxis, :]

    # member info only if reconstruction is an ensemble
    if reconstruction.shape[0] == 1:
        memberinfo = False

    # ===================================
    # Filter inappropriate constellations

    # return nan dataset if at least one nan is in any of the fields
    # or if one of the fields not above wet threshold (zero usually)
    if (np.isnan(reconstruction).any() or np.isnan(reference).any()) or (
        np.max(reconstruction) <= wet_thld or np.max(reference) <= wet_thld
    ):
        calculation_feasible = False

    else:
        # threshold: either field dependent or fixed
        if fixed_thld is None:
            # calculate thresholds
            R_high_rec = np.quantile(reconstruction[reconstruction > wet_thld], q=quantile)
            R_high_ref = np.quantile(reference[reference > wet_thld], q=quantile)
            
            # threshold factor
            thld_factor = np.max((thld_factor, 0.1 / R_high_ref))
            
            thld_rec = R_high_rec * thld_factor
            thld_ref = R_high_ref * thld_factor
        else:
            # take a fixed absolute threshold
            thld_rec = fixed_thld
            thld_ref = fixed_thld

        # return nan if thld is too high to allow for any feature
        if np.any(np.max(reconstruction, axis=(1, 2)) <= thld_rec) or np.any(
            np.max(reference, axis=(1, 2)) <= thld_ref
        ):
            calculation_feasible = False

    # ===================================
    # Actual calculations

    if calculation_feasible:

        # calculate maximum distance possible on grid
        maxDist = (
            (reconstruction.shape[1] - 1) ** 2 + (reconstruction.shape[2] - 1) ** 2
        ) ** 0.5

        # regional properties
        regionProps_rec = calc_region_properties(reconstruction, thld=thld_rec)
        regionProps_ref = calc_region_properties(reference, thld=thld_ref)

        # SAL / eSAL calculation
        eSAL = calc_eSAL(
            reconstruction, reference, regionProps_rec, regionProps_ref, maxDist
        )

        # SAL calculation of ensemble members
        if memberinfo:
            mSAL = calc_mSAL(
                reconstruction, reference, regionProps_rec, regionProps_ref, maxDist
            )

    # ===================================
    # Post-processing
    
    # store variables of interest (params) in a dictionary
    out_dict = {"time": time}
    
    if calculation_feasible:

        for p in params:
            if p in ["S", "A", "L", "L1"]:
                out_dict[p] = eSAL[p]
                if memberinfo:
                    out_dict["m%s" % p] = mSAL[p]
            if p == "thld":
                out_dict["%s_rec" % p] = thld_rec
                out_dict["%s_ref" % p] = thld_ref
            if p == "R_center":
                out_dict["%s_y_rec" % p] = regionProps_rec[p][..., 0]
                out_dict["%s_x_rec" % p] = regionProps_rec[p][..., 1]
                out_dict["%s_y_ref" % p] = regionProps_ref[p][..., 0]
                out_dict["%s_x_ref" % p] = regionProps_ref[p][..., 1]
            if p == "nF":
                out_dict["%s_rec" % p] = regionProps_rec[p]
                out_dict["%s_ref" % p] = regionProps_ref[p]

    else:

        for p in params:
            if p in ["S", "A", "L", "L1"]:
                out_dict[p] = np.nan
                if memberinfo:
                    out_dict["m%s" % p] = np.nan
            if p in ["thld", "nF"]:
                out_dict["%s_rec" % p] = np.nan
                out_dict["%s_ref" % p] = np.nan
            if p == "R_center":
                out_dict["%s_y_rec" % p] = np.nan
                out_dict["%s_x_rec" % p] = np.nan
                out_dict["%s_y_ref" % p] = np.nan
                out_dict["%s_x_ref" % p] = np.nan

    # ===================================
    # As Xarray Dataset

    # if variables should be given as xarray dataset
    if as_dataset:
        out_ds = build_dataset_timestep(
            out_dict, reconstruction.shape[0], reference.shape[0], calculation_feasible
        )
        out = out_ds
    else:
        out = out_dict

    return out


def build_dataset_timestep(out_dict, nfields_rec, nfields_ref, calculation_feasible):
    """
    Transform dictionary to xarray dataset.
    """
    
    out_ds = xr.Dataset({"time": out_dict["time"]})

    if calculation_feasible:
        for p in out_dict:
            if p in ["S", "A", "L", "L1", "thld_rec", "thld_ref"]:
                out_ds[p] = out_dict[p]
            if p in ["mS", "mA", "mL", "mL1", "R_center_y_rec", "R_center_x_rec", "nF_rec"]:
                out_ds[p] = ("nfields_rec", out_dict[p])
            if p in ["R_center_y_ref", "R_center_x_ref", "nF_ref"]:
                out_ds[p] = ("nfields_ref", out_dict[p])
    else:
        for p in out_dict:
            if p != "time":
                out_ds[p] = np.nan

    return out_ds


def SAL_timeseries(
    reconstruction,
    reference,
    t_array,
    yx_shift=None,
    thld_factor=1 / 15,
    quantile=0.95,
    fixed_thld=None,
    wet_thld=0,
    memberinfo=True,
    params=["S", "A", "L", "L1", "thld", "nF", "R_center"],
    workers=None,
):
    """
    Wrapper for timeseries calculation.

    Parameters:
    reconstruction, reference: xarray dataarrays with fields / ensembles and a "time" dimension.
    t_array: timeseries
    yx_shift: list / array [yshift, xshift], for coordinate shift. Needed only for centers of mass which are calculated on the fields assuming [0, ymax], [0, xmax] spatial extension.
    thld_factor, quantile, fixed_thld, wet_thld, memberinfo, params: passed on (see "SAL_timestep")
    workers: number of parallel jobs if calculated in parallel

    Returns:
    xarray dataset with parameters over time
    """

    if workers is not None:

        results = []
        for t in tqdm(t_array):

            results.append(
                dask.delayed(SAL_timestep)(
                    reconstruction.sel(time=t),
                    reference.sel(time=t),
                    time=t,
                )
            )

        sal = []
        for i in tqdm(range(int(len(t_array) / workers) + 1)):
            sal.extend(dask.compute(*results[workers * i : workers * (i + 1)]))

    else:
        sal = []
        for t in tqdm(t_array):

            sal.append(
                SAL_timestep(
                    reconstruction.sel(time=t),
                    reference.sel(time=t),
                    time=t,
                )
            )

    # concat along time
    if len(t_array) > 1:
        sal = xr.concat(sal, dim="time")

    # squeeze dataset (e.g. if reference is not an ensemble)
    sal = sal.squeeze()

    if yx_shift is not None:
        # bring to right grid position
        sal["R_center_y_rec"] = sal.R_center_y_rec + yx_shift[0]
        sal["R_center_x_rec"] = sal.R_center_x_rec + yx_shift[1]
        sal["R_center_y_ref"] = sal.R_center_y_ref + yx_shift[0]
        sal["R_center_x_ref"] = sal.R_center_x_ref + yx_shift[1]

    # add Euclidean distance
    sal["Q"] = (sal["S"] ** 2 + sal["A"] ** 2 + sal["L"] ** 2) ** 0.5

    return sal



#########################################################
# Functions to translate domain-wide SAL values to easily interpretable units


def A_to_percent(a):
    """
    Calculate under- and over-estimation in percent.
    """
    return 100 * (1 + 0.5 * a) / (1 - 0.5 * a)


def L1_to_km(a, ysz=900, xsz=700):
    """
    Calculate dislocation of total center of mass in grid units (not necessarily km).
    """
    maxDist = ((ysz - 1) ** 2 + (xsz - 1) ** 2) ** 0.5
    return a * maxDist
