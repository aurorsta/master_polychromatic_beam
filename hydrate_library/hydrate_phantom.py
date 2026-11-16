"""
Realistic 2D phantom of a hydrate-bearing sandstone core, and a simple CT forward model.

Builds on the sandstone_image() function in beam_hardening_basics.ipynb and adds
  1. brine whose salinity rises as water converts to hydrate
  2. a CO2 phase in the pores (liquid or gas)
  3. five hydrate morphologies (coating, filling, cementing, porous, patchy)
  4. phase volumes set by initial water saturation Sw0 and conversion fraction conv
  5. block averaging to medical-CT voxel size and the CT_wet / CT_dry saturation estimate
  6. an optional aluminium pressure sleeve
  7. Poisson noise that actually reaches the reconstruction
  8. optional polychromatic projection (for beam-hardening tests)

Units: lengths in mm, mu in 1/mm, energies in keV.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.transform import radon, iradon
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import disk
import xraylib as x

# Material label numbers used in the label map
OUTSIDE, QUARTZ, BRINE, HYDRATE, CO2, SLEEVE = 0, 1, 2, 3, 4, 5
NAMES = {OUTSIDE: "outside", QUARTZ: "quartz", BRINE: "brine",
         HYDRATE: "hydrate", CO2: "CO2", SLEEVE: "sleeve"}


# ----------------------------------------------------------------------------
# Attenuation
# ----------------------------------------------------------------------------
def brine_density(wt):
    """Approximate NaCl brine density [g/cm^3] at ~0-5 C. wt is the mass fraction of NaCl."""
    return 1.0 + 0.71 * wt


def mu_compound(formula, density, energy_kev):
    """Linear attenuation [1/mm] from xraylib (total, incl. coherent scattering)."""
    return x.CS_Total_CP(formula, energy_kev) * density / 10


def mu_brine(wt, energy_kev):
    """Linear attenuation [1/mm] of NaCl brine with salt mass fraction wt."""
    mass_mu = wt * x.CS_Total_CP("NaCl", energy_kev) + (1 - wt) * x.CS_Total_CP("H2O", energy_kev)
    return mass_mu * brine_density(wt) / 10


def material_mus(energy_kev, salinity, n_h=6.08, rho_hydrate=1.09, rho_co2=0.07):
    """
    mu [1/mm] for every label at one energy.
    salinity    : NaCl mass fraction of the brine that is LEFT in the pores
    n_h         : hydration number (6.08 from the medical CT paper, ideal sI is 5.75)
    rho_co2     : CO2 density [g/cm^3]. ~0.07 for gas at 30 bar and 1 C (default),
                  ~0.9 for liquid CO2, ~0.0012 for air in a dry reference scan
    """
    return {OUTSIDE: 0.0,
            QUARTZ:  mu_compound("SiO2", 2.65, energy_kev),
            BRINE:   mu_brine(salinity, energy_kev),
            HYDRATE: mu_compound(f"CO2(H2O){n_h}", rho_hydrate, energy_kev),
            CO2:     mu_compound("CO2", rho_co2, energy_kev),
            SLEEVE:  mu_compound("Al", 2.70, energy_kev)}


# ----------------------------------------------------------------------------
# Grains
# ----------------------------------------------------------------------------
def make_grains(n_pixel, pixel_size, core_radius, porosity=0.23, grain_radius=0.12, seed=None):
    """Random overlapping quartz disks inside a round core until the porosity is reached."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n_pixel, :n_pixel]
    c = (n_pixel - 1) / 2
    core = np.hypot(xx - c, yy - c) <= core_radius / pixel_size

    grain = np.zeros((n_pixel, n_pixel), bool)
    r_px = grain_radius / pixel_size
    while (~grain[core]).mean() > porosity:
        x0, y0 = rng.uniform(0, n_pixel, 2)
        r = r_px * rng.uniform(0.7, 1.3)
        # only update a small window around the new grain (much faster than the full grid)
        y_lo, y_hi = int(max(y0 - r - 1, 0)), int(min(y0 + r + 2, n_pixel))
        x_lo, x_hi = int(max(x0 - r - 1, 0)), int(min(x0 + r + 2, n_pixel))
        sub_y, sub_x = yy[y_lo:y_hi, x_lo:x_hi], xx[y_lo:y_hi, x_lo:x_hi]
        grain[y_lo:y_hi, x_lo:x_hi] |= (sub_x - x0)**2 + (sub_y - y0)**2 <= r**2
    grain &= core
    pore = core & ~grain
    return grain, pore, core


# ----------------------------------------------------------------------------
# Hydrate placement: each morphology is just an ORDER in which pore pixels turn into hydrate
# ----------------------------------------------------------------------------
def split_pores(pore, min_distance=4):
    """Split the pore space into individual pores (watershed on the distance map)."""
    d = ndi.distance_transform_edt(pore)
    peaks = peak_local_max(d, min_distance=min_distance, labels=pore)
    markers = np.zeros(pore.shape, int)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return watershed(-d, markers, mask=pore)


def hydrate_order(morphology, grain, pore, pixel_size, rng):
    """
    Return the flat indices of pore pixels in the order they become hydrate.
      coating   : nearest to grains first (thin layer on the grain surfaces)
      filling   : whole pores, one at a time (non-porous, pathway B)
      cementing : narrow necks and grain contacts first
      porous    : a shell around CO2 in the pore centres (pathway A)
      patchy    : everything near a random point first (hydrate only in one region)
    """
    idx = np.flatnonzero(pore)
    d_grain = ndi.distance_transform_edt(~grain).flat[idx]          # distance to nearest grain [px]

    if morphology == "coating":
        return idx[np.argsort(d_grain, kind="stable")]

    if morphology == "filling":
        pores = split_pores(pore).flat[idx]
        pore_ids = rng.permutation(np.unique(pores[pores > 0]))
        rank = np.empty(pores.max() + 1)
        rank[pore_ids] = np.arange(len(pore_ids))
        return idx[np.lexsort((d_grain, rank[pores]))]               # pore by pore, each filled fully

    if morphology == "cementing":
        # a pixel's rank is the smallest closing radius that turns it solid: contacts close first
        closing_rank = np.full(idx.size, np.inf)
        for k in range(1, 30):
            closed = ndi.binary_closing(grain, structure=disk(k), border_value=1).flat[idx]
            closing_rank[(closing_rank == np.inf) & closed] = k
        return idx[np.lexsort((d_grain, closing_rank))]

    if morphology == "porous":
        # caller puts CO2 in the pore centres first; hydrate then takes the next layer outwards
        return idx[np.argsort(-d_grain, kind="stable")]

    if morphology == "patchy":
        rows, cols = np.unravel_index(idx, pore.shape)
        y0, x0 = rows[rng.integers(idx.size)], cols[rng.integers(idx.size)]
        return idx[np.argsort(np.hypot(rows - y0, cols - x0), kind="stable")]

    raise ValueError(f"unknown morphology {morphology!r}")


# ----------------------------------------------------------------------------
# Full sample
# ----------------------------------------------------------------------------
def hydrate_core(n_pixel=512, pixel_size=0.01, core_radius=2.2, sleeve_thickness=0.3,
                 porosity=0.23, grain_radius=0.12,
                 Sw0=0.5, conv=0.8, expansion=1.26, salinity0=0.035,
                 morphology="coating", seed=None):
    """
    Label map of a hydrate-bearing core.

    Sw0        : water saturation before hydrate forms
    conv       : fraction of that water converted to hydrate
    expansion  : hydrate volume / water volume it came from (1.26, Almenningen et al. 2021)
    salinity0  : initial NaCl mass fraction. Salt is excluded from hydrate, so the remaining
                 brine becomes saltier: salinity = salinity0 / (1 - conv) (capped at 0.26, halite saturation)

    Returns labels (int array), info (dict with true saturations and salinity).
    """
    rng = np.random.default_rng(seed)
    grain, pore, core = make_grains(n_pixel, pixel_size, core_radius, porosity, grain_radius, seed)
    n_pore = pore.sum()

    # Phase volumes in pore pixels
    S_H = expansion * conv * Sw0
    S_w = (1 - conv) * Sw0
    if S_H + S_w > 1:
        raise ValueError("hydrate + brine exceeds the pore volume; lower Sw0 or conv")
    n_hyd, n_wat = int(round(S_H * n_pore)), int(round(S_w * n_pore))
    n_co2 = n_pore - n_hyd - n_wat

    labels = np.zeros(pore.shape, int)
    labels[grain] = QUARTZ
    free = pore.copy()

    # Porous hydrate: CO2 bubbles sit in the pore centres and the hydrate shell grows around them
    if morphology == "porous":
        idx = np.flatnonzero(free)
        d = ndi.distance_transform_edt(~grain).flat[idx]
        co2_idx = idx[np.argsort(-d, kind="stable")][:n_co2]
        labels.flat[co2_idx] = CO2
        free.flat[co2_idx] = False
        n_co2 = 0

    # Hydrate
    order = hydrate_order(morphology, grain, free, pixel_size, rng)
    labels.flat[order[:n_hyd]] = HYDRATE
    free.flat[order[:n_hyd]] = False

    # Water-wet rock: brine in the corners near grains, CO2 in what is left (pore centres)
    idx = np.flatnonzero(free)
    d = ndi.distance_transform_edt(~grain).flat[idx]
    idx = idx[np.argsort(d, kind="stable")]
    labels.flat[idx[:n_wat]] = BRINE
    labels.flat[idx[n_wat:]] = CO2

    # Pressure sleeve
    if sleeve_thickness > 0:
        yy, xx = np.mgrid[:n_pixel, :n_pixel]
        c = (n_pixel - 1) / 2
        r = np.hypot(xx - c, yy - c) * pixel_size
        labels[(r > core_radius) & (r <= core_radius + sleeve_thickness)] = SLEEVE

    salinity = min(salinity0 / max(1 - conv, 1e-9), 0.26)
    info = {"S_H": (labels == HYDRATE).sum() / n_pore,
            "S_w": (labels == BRINE).sum() / n_pore,
            "S_co2": (labels == CO2).sum() / n_pore,
            "salinity": salinity, "conv": conv, "Sw0": Sw0,
            "pore": pore, "core": core}
    return labels, info


def mu_image(labels, mus):
    """Turn a label map into a mu map [1/mm]."""
    img = np.zeros(labels.shape)
    for lab, mu in mus.items():
        img[labels == lab] = mu
    return img


# ----------------------------------------------------------------------------
# Forward model and reconstruction
# ----------------------------------------------------------------------------
def scan(labels, pixel_size, theta, energies_kev, weights, N0=1e5, mus_kwargs=None, rng=None):
    """
    Simulated CT scan. Monochromatic if energies_kev has one value.
    Projects each material mask once (path length per ray), then applies Beer-Lambert per
    energy bin and sums over the spectrum, so beam hardening comes out by itself.
    Returns the noisy sinogram -ln(I/I0) / pixel_size, ready for iradon.
    """
    rng = np.random.default_rng(rng)
    mus_kwargs = mus_kwargs or {}
    weights = np.asarray(weights, float) / np.sum(weights)

    # path length through each material [mm] for every ray
    paths = {lab: radon((labels == lab).astype(float), theta=theta) * pixel_size
             for lab in np.unique(labels) if lab != OUTSIDE}

    I = 0.0
    for E, w in zip(np.atleast_1d(energies_kev), weights):
        mus = material_mus(float(E), **mus_kwargs)
        I = I + w * np.exp(-sum(mus[lab] * L for lab, L in paths.items()))

    counts = np.maximum(rng.poisson(N0 * I), 1)                 # Poisson noise, avoid log(0)
    return -np.log(counts / N0) / pixel_size


def reconstruct(sinogram, theta):
    return iradon(sinogram, theta=theta, filter_name="ramp")


def block_average(img, block):
    """Average block x block pixels, e.g. 10 um pixels -> 0.3 mm medical-CT voxels (block = 30)."""
    n = (img.shape[0] // block) * block
    return img[:n, :n].reshape(n // block, block, n // block, block).mean(axis=(1, 3))


def medical_ct_saturation(rec_exp, rec_dry, rec_wet):
    """S_{w+H} = (CT_exp - CT_dry) / (CT_wet - CT_dry), as in the medical CT paper (their eq. 1)."""
    num, den = rec_exp - rec_dry, rec_wet - rec_dry
    # voxels outside the reconstruction circle are exactly 0 in all three scans -> NaN, not a 0/0 warning
    return np.divide(num, den, out=np.full_like(den, np.nan, dtype=float), where=den != 0)
