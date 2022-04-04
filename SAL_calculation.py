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
    thld: threshold to discriminate rainfall objects from surrounding.
    
    Returns:
    dictionary of regional properties.
    """

    # mask used for properties of total domain
    mask = np.ones(fields.shape[1:]).astype(int)

    # initialize output variables (volume, total center of mass,
    # feature distances, number of features)
    V_member = []
    tcm_member = []
    fDist_member = []
    nF_member = []
    for member in fields:

        # total center of mass
        domain = regionprops(label_image=mask, intensity_image=member)
        tcm = np.array(domain[0].weighted_centroid)
        tcm_member.append(tcm)

        # ===================================
        # feature properties

        # identify features
        apply_thld = np.copy(member)
        apply_thld[apply_thld <= thld] = 0
        l, m = snd.label(apply_thld)
        features = regionprops(label_image=l, intensity_image=member)

        # local center of mass, area, mean, max
        area_, mean_, max_, lcm_ = [], [], [], []
        # n_ = 0
        for f in features:
            area_.append(f.area)
            mean_.append(f.mean_intensity)
            max_.append(f.max_intensity)
            lcm_.append(f.weighted_centroid)
            # n_ += 1

        # sums of rain in features
        R_n = np.array(area_) * np.array(mean_)

        # ===================================
        # calculations that reduce feature dimension

        # volume normalized by domain sum of rain
        V_member.append((R_n**2 / np.array(max_)).sum() / R_n.sum())

        # weightet distances of center of masses to
        # total center of mass normalized by domain sum of rain
        fDist_member.append((R_n * calc_dist(tcm, np.array(lcm_))).sum() / R_n.sum())

        # number of features
        nF_member.append(len(features))

    regionProps = dict(
        V=np.array(V_member),
        tcm=np.array(tcm_member),
        fDist=np.array(fDist_member),
        nF=np.array(nF_member),
    )

    return regionProps


def calc_mSAL(a, b, rProp_a, rProp_b, maxDist):
    """
    Calculate SAL parameters of the individual ensemble members (mSAL).
    
    Parameters:
    a: simulation field / ensemble
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
    L1 = calc_dist(rProp_a["tcm"], np.mean(rProp_b["tcm"], axis=0)) / maxDist
    L2 = 2 * np.abs(rProp_a["fDist"] - np.mean(rProp_a["fDist"])) / maxDist
    L = L1 + L2

    return dict(S=S, A=A, L=L, L1=L1)


def calc_eSAL(a, b, rProp_a, rProp_b, maxDist):
    """
    Calculate eSAL parameters of the whole ensemble.
    
    Parameters:
    a: simulation field / ensemble
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
    tcm_a = np.mean(rProp_a["tcm"], axis=0)
    tcm_b = np.mean(rProp_b["tcm"], axis=0)
    eL1 = calc_dist(tcm_a, tcm_b) / maxDist
    eL2 = 2 * calc_crps(rProp_a["fDist"] / maxDist, rProp_b["fDist"] / maxDist)
    eL = eL1 + eL2

    return dict(S=eS, A=eA, L=eL, L1=eL1)


def SAL_timestep(
    simulation,
    reference,
    time="xxx",
    thld_factor=1 / 15,
    quantile=0.95,
    fixed_thld=None,
    wet_thld=0,
    memberinfo=True,
    params=["S", "A", "L", "L1", "thld", "nF", "tcm"],
    as_dataset=True,
):

    """
    Wrapper for calculating (e)SAL parameters (and additional parameters) for one simulation (ensemble or single field) and reference (ensemble or single field). 
    
    Parameters:
    simulation, reference: Both can be a single field (shape = (ysize, xsize))
        or an ensemble (shape = (n_members, ysize, xsize)). In the latter case "SAL" is acutally "eSAL". Input fields can be numpy arrays or xarray dataarrays.
        
    time: index for the output (usually time but could be any string or number)
        
    thld_factor, quantile: used to define threshold contour, i.e. the minimum intensity required to belong to a feature.
        
    fixed_thld: give threshold directly without calculating via thld_factor and quantile
        
    wet_thld: value above which a pixel is considered wet
    
    memberinfo: In case simulation is an ensemble this returns parameters for the individual members also
        
    params: list of parameters that shall be returned
        
    as_dataset: if true returns an xarray dataset, otherwise a dictionary
        
    Returns:
        Xarray Dataset or dictionary of parameters
    """

    # ===================================
    # Pre-processing

    # if sim and ref are given as xarray dataarrays transform them to numpy arrays
    simulation = trans_xr_np(simulation)
    reference = trans_xr_np(reference)

    # assert right shape of input (either 2D: single field, or 3D: ensemble)
    assert (len(simulation.shape) in [2, 3]) and (
        len(reference.shape) in [2, 3]
    ), "Input must be 2- or 3-dimensional."

    # set flag whether SAL calculation can be conducted
    calculation_feasible = True

    # if single field and no ensemble is given change shape
    # (single fields are treated as ensemble of one member)
    if len(simulation.shape) == 2:
        simulation = simulation[np.newaxis, :]
    if len(reference.shape) == 2:
        reference = reference[np.newaxis, :]

    # member info only if simulation is an ensemble
    if simulation.shape[0] == 1:
        memberinfo = False

    # ===================================
    # Filter inappropriate constellations

    # return nan dataset if at least one nan is in any of the fields
    # or if one of the fields not above wet threshold (zero usually)
    if (np.isnan(simulation).any() or np.isnan(reference).any()) or (
        np.max(simulation) <= wet_thld or np.max(reference) <= wet_thld
    ):
        calculation_feasible = False

    else:
        # threshold: either field dependent or fixed
        if fixed_thld is None:
            # calculate thresholds
            R_sim_high = np.quantile(simulation[simulation > wet_thld], q=quantile)
            R_ref_high = np.quantile(reference[reference > wet_thld], q=quantile)
            
            # NEEDS TO BE MORE GENERAL
            thld_factor = np.max((thld_factor, 0.1 / R_ref_high))
            
            thld_sim = R_sim_high * thld_factor
            thld_ref = R_ref_high * thld_factor
        else:
            # take a fixed absolute threshold
            thld_sim = fixed_thld
            thld_ref = fixed_thld

        # return nan if thld is too high to allow for any feature
        if np.any(np.max(simulation, axis=(1, 2)) <= thld_sim) or np.any(
            np.max(reference, axis=(1, 2)) <= thld_ref
        ):
            calculation_feasible = False

    # ===================================
    # Actual calculations

    if calculation_feasible:

        # calculate maximum distance possible on grid
        maxDist = (
            (simulation.shape[1] - 1) ** 2 + (simulation.shape[2] - 1) ** 2
        ) ** 0.5

        # regional properties
        regionProps_sim = calc_region_properties(simulation, thld=thld_sim)
        regionProps_ref = calc_region_properties(reference, thld=thld_ref)

        # SAL / eSAL calculation
        eSAL = calc_eSAL(
            simulation, reference, regionProps_sim, regionProps_ref, maxDist
        )

        # SAL calculation of ensemble members
        if memberinfo:
            mSAL = calc_mSAL(
                simulation, reference, regionProps_sim, regionProps_ref, maxDist
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
                out_dict["%s_sim" % p] = thld_sim
                out_dict["%s_ref" % p] = thld_ref
            if p == "tcm":
                out_dict["%s_y_sim" % p] = regionProps_sim[p][..., 0]
                out_dict["%s_x_sim" % p] = regionProps_sim[p][..., 1]
                out_dict["%s_y_ref" % p] = regionProps_ref[p][..., 0]
                out_dict["%s_x_ref" % p] = regionProps_ref[p][..., 1]
            if p == "nF":
                out_dict["%s_sim" % p] = regionProps_sim[p]
                out_dict["%s_ref" % p] = regionProps_ref[p]

    else:

        for p in params:
            if p in ["S", "A", "L", "L1"]:
                out_dict[p] = np.nan
                if memberinfo:
                    out_dict["m%s" % p] = np.nan
            if p in ["thld", "nF"]:
                out_dict["%s_sim" % p] = np.nan
                out_dict["%s_ref" % p] = np.nan
            if p == "tcm":
                out_dict["%s_y_sim" % p] = np.nan
                out_dict["%s_x_sim" % p] = np.nan
                out_dict["%s_y_ref" % p] = np.nan
                out_dict["%s_x_ref" % p] = np.nan

    # ===================================
    # As Xarray Dataset

    # if variables should be given as xarray dataset
    if as_dataset:
        out_ds = build_dataset_timestep(
            out_dict, simulation.shape[0], reference.shape[0], calculation_feasible
        )
        out = out_ds
    else:
        out = out_dict

    return out


def build_dataset_timestep(out_dict, nfields_sim, nfields_ref, calculation_feasible):
    """
    Transform dictionary to xarray dataset.
    """
    
    out_ds = xr.Dataset({"time": out_dict["time"]})

    if calculation_feasible:
        for p in out_dict:
            if p in ["S", "A", "L", "L1", "thld_sim", "thld_ref"]:
                out_ds[p] = out_dict[p]
            if p in ["mS", "mA", "mL", "mL1", "tcm_y_sim", "tcm_x_sim", "nF_sim"]:
                out_ds[p] = ("nfields_sim", out_dict[p])
            if p in ["tcm_y_ref", "tcm_x_ref", "nF_ref"]:
                out_ds[p] = ("nfields_ref", out_dict[p])
    else:
        for p in out_dict:
            if p != "time":
                out_ds[p] = np.nan

    return out_ds


def SAL_timeseries(
    simulation,
    reference,
    t_array,
    yx_shift=None,
    thld_factor=1 / 15,
    quantile=0.95,
    fixed_thld=None,
    wet_thld=0,
    memberinfo=True,
    params=["S", "A", "L", "L1", "thld", "nF", "tcm"],
    workers=None,
):
    """
    Wrapper for timeseries calculation.

    Parameters:
    simulation, reference: xarray dataarrays with fields / ensembles and a "time" dimension.
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
                    simulation.sel(time=t),
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
                    simulation.sel(time=t),
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
        sal["tcm_y_sim"] = sal.tcm_y_sim + yx_shift[0]
        sal["tcm_x_sim"] = sal.tcm_x_sim + yx_shift[1]
        sal["tcm_y_ref"] = sal.tcm_y_ref + yx_shift[0]
        sal["tcm_x_ref"] = sal.tcm_x_ref + yx_shift[1]

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
