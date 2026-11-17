"""
2D CT phantom of a sandstone core with water, CO2 hydrate and CO2 in the pores.

Change the four inputs below and run the file. Units: mm, 1/mm, keV.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.transform import radon, iradon
import xraylib as xrl

# ---- inputs
# the core starts fully water saturated; hydrate and CO2 then take part of the pore space
S_H = 0.45                                           # hydrate saturation (fraction of pore volume)
S_CO2 = 0.05                                         # free CO2 saturation
S_w = 1 - S_H - S_CO2                               # water left over
energy = 120.0                                      # beam energy [keV]
N0 = None                                            # photons per detector pixel per projection (None = no noise)
theta = np.linspace(0, 180, 360, endpoint=False)    # projection angles [deg]

# ---- fixed sample
n_pixel, pixel_size = 512, 0.01                     # 10 um pixels -> 5.12 mm image
core_radius, grain_radius, porosity = 2.4, 0.12, 0.23

QUARTZ, WATER, HYDRATE, CO2 = 1, 2, 3, 4
MATERIALS = {QUARTZ:  ("SiO2", 2.65),               # (formula, density [g/cm^3])
             WATER:   ("H2O", 1.00),
             HYDRATE: ("CO2(H2O)6.08", 1.09),       # n_h and density from the medical CT paper
             CO2:     ("CO2", 0.07)}                # CO2 gas at ~30 bar, 1 C

# ---- hydrate growth
EXPANSION = 1.26    # hydrate volume / volume of water it consumed (Almenningen et al. 2021)
N_NUCLEI = 50       # nucleation sites on the water-CO2 interface
LATERAL = 5         # the hydrate film spreads this many times faster along the interface than it thickens


def make_core(S_H, S_CO2, seed=0):
    """
    Label map, built in the order things happen in the experiment:
      1. sandstone: random overlapping quartz grains, pores 100 % water
      2. CO2 injected into the water-wet rock: CO2 pushes water out of the widest pores first
         (capillary drainage), water stays in the narrow throats, grain contacts and corners
      3. hydrate nucleates at a few random points on the water-CO2 interface, spreads quickly
         along the interface as a film, then thickens slowly. It mostly eats into the water,
         and since it takes up 26 % more volume than that water, it also grows into the CO2.
    S_H and S_CO2 are the saturations at the end; water fills the rest.
    """
    if S_H + S_CO2 > 1 or min(S_H, S_CO2) < 0:
        raise ValueError("need S_H, S_CO2 >= 0 and S_H + S_CO2 <= 1")
    S_w = 1 - S_H - S_CO2
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n_pixel, :n_pixel]
    c = (n_pixel - 1) / 2
    core = np.hypot(xx - c, yy - c) <= core_radius / pixel_size

    grain = np.zeros(core.shape, bool)
    while (~grain[core]).mean() > porosity:
        x0, y0 = rng.uniform(0, n_pixel, 2)
        r = grain_radius / pixel_size * rng.uniform(0.7, 1.3)
        grain |= (xx - x0)**2 + (yy - y0)**2 <= r**2
    grain &= core

    pore = core & ~grain

    # 2. capillary drainage. Pore size of a pixel = radius of the largest disk that fits in the
    #    pore space and covers the pixel (morphological opening, done with distance transforms)
    d_wall = ndi.distance_transform_edt(pore)
    size = np.zeros(pore.shape)
    for r in range(1, int(d_wall.max()) + 1):
        size[ndi.distance_transform_edt(d_wall < r) <= r] = r
    idx = np.flatnonzero(pore)
    idx = idx[np.lexsort((-d_wall.flat[idx], -size.flat[idx]))]   # widest pores first, centres first
    n_h, n_w = round(S_H * idx.size), round(S_w * idx.size)
    n_w0 = n_w + round(n_h / EXPANSION)                            # water before hydrate formed
    labels = np.where(grain, QUARTZ, 0)
    labels.flat[idx[:idx.size - n_w0]] = CO2
    labels.flat[idx[idx.size - n_w0:]] = WATER

    # 3. hydrate growth: arrival time of the growth front = distance from the interface (thickening)
    #    + distance from the nearest nucleus / LATERAL (fast film spreading along the interface)
    if n_h > 0:
        water, co2 = labels == WATER, labels == CO2
        d_int = np.where(water, ndi.distance_transform_edt(~co2), ndi.distance_transform_edt(~water))
        interface = np.flatnonzero(water & ndi.binary_dilation(co2))
        not_nucleus = np.ones(pore.shape, bool)
        not_nucleus.flat[rng.choice(interface, min(N_NUCLEI, interface.size), replace=False)] = False
        t = d_int + ndi.distance_transform_edt(not_nucleus) / LATERAL
        for phase, n in ((water, n_w0 - n_w), (co2, n_h - (n_w0 - n_w))):
            p = np.flatnonzero(phase)
            labels.flat[p[np.argsort(t.flat[p], kind="stable")[:n]]] = HYDRATE
    return labels


def mu_map(labels, energy):
    """Linear attenuation [1/mm] for each pixel at one beam energy."""
    mu = np.zeros(labels.shape)
    for lab, (formula, rho) in MATERIALS.items():
        mu[labels == lab] = xrl.CS_Total_CP(formula, float(energy)) * rho / 10
    return mu


def scan(mu, theta, N0):
    """
    Detector measurement: Beer-Lambert with Poisson photon noise, then back to line integrals.
    radon() sums mu over pixels, so multiply by pixel_size to get the dimensionless line integral.
    """
    line_integral = radon(mu, theta=theta) * pixel_size
    if N0 is None:
        return line_integral / pixel_size
    rng = np.random.default_rng(1)
    counts = np.maximum(rng.poisson(N0 * np.exp(-line_integral)), 1)   # avoid log(0)
    return -np.log(counts / N0) / pixel_size


labels = make_core(S_H, S_CO2)
mu = mu_map(labels, energy)
sinogram = scan(mu, theta, N0)
recon = iradon(sinogram, theta=theta, filter_name="ramp")

# mean reconstructed value per phase (eroded masks, so edges and partial pixels don't count)
for lab, name in ((QUARTZ, "quartz"), (WATER, "water"), (HYDRATE, "hydrate"), (CO2, "CO2")):
    m = ndi.binary_erosion(labels == lab, iterations=2)
    if m.any():
        print(f"{name:8s} true mu {mu[labels == lab].mean():.5f}   recon {recon[m].mean():.5f} "
              f"+- {recon[m].std():.5f} 1/mm")

ext = (0, n_pixel * pixel_size, n_pixel * pixel_size, 0)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()                                    # axes[0] ... axes[5], row by row
cmap = plt.matplotlib.colors.ListedColormap(["white", "tan", "tab:blue", "tab:red", "lightgrey"])

# 1. sample
axes[0].imshow(labels, cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest", extent=ext)
axes[0].set_title(f"Sample: $S_H$={S_H}, $S_{{CO_2}}$={S_CO2}, $S_w$={S_w:.2f}")
axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in (1, 2, 3, 4)],
               labels=["quartz", "water", "hydrate", "CO$_2$"], loc="upper right", fontsize=8)

# 2. sinogram
axes[1].imshow(sinogram, cmap="Greys_r", aspect="auto", extent=(theta[0], theta[-1], sinogram.shape[0], 0))
axes[1].set_title(f"Sinogram, {len(theta)} projections")
axes[1].set_xlabel("angle [deg]")

# 3. reconstruction, narrow window around water and hydrate
mu_w, mu_h = mu[labels == WATER].mean(), mu[labels == HYDRATE].mean()
contrast = mu_h - mu_w
im = axes[2].imshow(recon, cmap="Greys_r", vmin=mu_w - 0.5 * contrast, vmax=mu_h + 0.5 * contrast, extent=ext)
axes[2].set_title(f"Reconstruction at {energy:g} keV\nnarrow window (water vs hydrate)")
fig.colorbar(im, ax=axes[2], label=r"$\mu$ [1/mm]")

# 4. error map, colour scale = +- hydrate-water contrast
pore = np.isin(labels, [WATER, HYDRATE, CO2])
error = recon - mu
rms_pore = np.sqrt(np.mean(error[pore]**2))
im = axes[3].imshow(error, cmap="Greys_r", vmin=-contrast, vmax=contrast, extent=ext)
axes[3].set_title(f"Reconstruction error\nRMS in pores: {rms_pore:.1e} /mm")
fig.colorbar(im, ax=axes[3], label=r"$\Delta\mu$ [1/mm]")

# 5. contrast between hydrate and water as a function of photon energy
energies = np.linspace(15, 150, 200)                                       # [keV]
mu_curve = {lab: np.array([xrl.CS_Total_CP(MATERIALS[lab][0], E) * MATERIALS[lab][1] / 10 for E in energies])
            for lab in (WATER, HYDRATE)}
diff = mu_curve[HYDRATE] - mu_curve[WATER]                                 # absolute contrast [1/mm]
rel = 100 * diff / mu_curve[WATER]                                         # relative contrast [%]
ax = axes[4]
ax.plot(energies, rel, color="tab:red")
ax.set_xlabel("Photon energy [keV]")
ax.set_ylabel(r"Relative contrast $(\mu_H - \mu_w)/\mu_w$ [%]", color="tab:red")
ax.set_ylim(0, 1.2 * rel.max())
ax2 = ax.twinx()
ax2.semilogy(energies, diff, color="tab:blue")
ax2.set_ylabel(r"Absolute contrast $\mu_H - \mu_w$ [1/mm]", color="tab:blue")
ax.axvline(energy, color="grey", ls="--", lw=1)
ax.set_title(f"Hydrate vs water contrast\n(dashed line: {energy:g} keV used above)")

# 6. reconstruction error for several energies
energies_test = np.arange(20, 151, 10)                                     # [keV]
rms_all = []
for E in energies_test:
    mu_test = mu_map(labels, E)
    rec_test = iradon(radon(mu_test, theta=theta), theta=theta, filter_name="ramp")
    rms_all.append(np.sqrt(np.mean((rec_test - mu_test)**2)))
axes[5].semilogy(energies_test, rms_all, "o-")
axes[5].axvline(energy, color="grey", ls="--", lw=1)
axes[5].set_xlabel("Photon energy [keV]")
axes[5].set_ylabel("RMS reconstruction error [1/mm]")
axes[5].set_title("Reconstruction error vs photon energy")

fig.tight_layout()
plt.show()