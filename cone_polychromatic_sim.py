# GPU dependent
import tigre
import tigre.algorithms as algs
from xpecgen import xpecgen as xg
from samples_2D import Sample2D, NAMES, LABEL, mu
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import os
from pathlib import Path
from datetime import datetime
import matplotlib
ON_CLUSTER = "SLURM_JOB_ID" in os.environ
if ON_CLUSTER:
    matplotlib.use("Agg")                  # no screen on the cluster
import matplotlib.pyplot as plt

# One folder per run, inside the repo: figures/2026-11-24_1433/
FIG_DIR = Path(__file__).resolve().parent / "figures" / datetime.now().strftime("%Y-%m-%d_%H%M")


def finish(fig, name):
    """Save on the cluster, show on the PC."""
    if ON_CLUSTER:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def energy_spectrum(kVp=100, filter_mm_Al=1.0, n_points=50):
    """Tungsten tube spectrum. Returns energies E [keV] and photons per bin S."""
    # Check these calls against the xpecgen README
    spec = xg.calculate_spectrum(kVp, 12, 3, n_points, epsrel=0.5, monitor=None, z=74)   # kVp, anode angle, phi
    spec.attenuate(filter_mm_Al / 10, xg.get_mu(13))                                     # Al filter, depth in cm
    E, S = spec.get_points()
    E, S = np.asarray(E), np.asarray(S)
    keep = E > 1                                                                         # drop the empty lowest bins
    return E[keep], S[keep]


def simulation(sample, E, S, pixel_size, theta_max=2 * np.pi, n_angles=720, N0=None, seed=0):
    # 1. Geometry and angles for fan beam 
    n = sample.n_pixel
    geo = tigre.geometry()
    # Distances
    geo.DSD = 200  # Distance Source Detector      (mm)  <- use your scanner's values
    geo.DSO = 50   # Distance Source Origin        (mm)
    mag = geo.DSD / geo.DSO
    # Image parameters
    geo.nVoxel = np.array([1, n, n])                       # number of voxels              (vx)
    geo.dVoxel = np.array([pixel_size] * 3)                # size of each voxel            (mm)
    geo.sVoxel = geo.nVoxel * geo.dVoxel                   # total size of the image       (mm)
    # Detector parameters
    geo.nDetector = np.array([1, int(1.2 * n)])            # number of pixels, with margin (px)
    geo.dDetector = np.array([geo.dVoxel[0], pixel_size * mag])  # size of each pixel      (mm)
    geo.sDetector = geo.nDetector * geo.dDetector          # total size of the detector    (mm)
    # Offsets
    geo.offOrigin = np.array([0, 0, 0])                    # Offset of image from origin   (mm)
    geo.offDetector = np.array([0, 0])                     # Offset of Detector            (mm)

    geo.mode = "cone"
    angles = np.linspace(0, theta_max, n_angles, endpoint=False)   # radians

    # 2. Projections, polychromatic
    # Path length [mm] of each material along every ray: one forward projection per material
    L = {name: tigre.Ax(m[None].astype(np.float32), geo, angles) for name, m in sample.masks().items()}
    mus = {name: mu(name, E) for name in L}                        # mu(E) [1/mm] for each material
    w = S / S.sum()
    I = np.zeros_like(next(iter(L.values())))                      # I/I0 on the detector
    for k in range(len(E)):
        I += w[k] * np.exp(-sum(mus[name][k] * L[name] for name in L))   # Beer-Lambert per energy
    if N0 is not None:
        rng = np.random.default_rng(seed)
        I = np.maximum(rng.poisson(N0 * I), 1) / N0                # Poisson noise on photon counts
    proj = (-np.log(I)).astype(np.float32)

    # 3. Reconstruct
    img_fdk = algs.fdk(proj, geo, angles)[0]   # cone beam filtered backprojection, (n, n)
    return proj, img_fdk, I, angles


def simu_plots(img, proj, recon, I, n_pixel, pixel_size, angles, tag=""):
    # TIGRE output (n_angles, 1, n_det) -> (n_det, n_angles), same layout as radon
    sino = proj[:, 0, :].T
    I = I[:, 0, :].T
    theta = np.degrees(angles)                                     # [deg]

    L_img = n_pixel * pixel_size
    ext = (0, L_img, L_img, 0)                                     # image extent [mm]
    dtheta = theta[1] - theta[0]                                   # angle step [deg]
    det_len = sino.shape[0] * pixel_size                           # detector length, scaled to the sample [mm]
    ext_sino = (-dtheta / 2, theta[-1] + dtheta / 2, 0, det_len)   # sinogram extent [deg, mm]

    # Original image and sinogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    ax1.set_title("Original")
    im1 = ax1.imshow(img, cmap="Greys_r", vmin=0, vmax=img.max(), extent=ext)
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    fig.colorbar(im1, ax=ax1, label=r"$\mu$ [1/mm]")

    ax2.set_title("Radon transform\n(Sinogram)")
    ax2.set_xlabel("Projection angle [deg]")
    ax2.set_ylabel("Detector position [mm]")
    im2 = ax2.imshow(sino, cmap="Greys_r", extent=ext_sino,       # no * pixel_size
                     origin="lower", aspect="auto")
    fig.colorbar(im2, ax=ax2, label=r"$\int \mu \, dl$")

    fig.tight_layout()
    finish(fig, f"{tag}_sinogram")

    # Intensity on the detector for a few angles
    fig, ax = plt.subplots(figsize=(6, 4.5))
    det = np.arange(I.shape[0]) * pixel_size
    for k in (0, len(theta) // 4, len(theta) // 2, len(theta) - 1):
        ax.plot(det, I[:, k], label=f"{theta[k]:.0f}°")
    ax.set_title("Intensity on the detector")
    ax.set_xlabel("Detector position [mm]")
    ax.set_ylabel("$I/I_0$")
    # ax.set_yscale("log")                     # log scale shows low intensities better
    ax.legend()
    fig.tight_layout()
    finish(fig, f"{tag}_detector_intensity")

    # Reconstruction and error
    error = recon - img
    lim = np.abs(error).max()                  # symmetric colour scale around zero
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)

    im1 = ax1.imshow(recon, cmap="Greys_r", vmin=0, vmax=img.max(), extent=ext)
    ax1.set_title("Reconstruction\nFiltered back projection")
    fig.colorbar(im1, ax=ax1, label=r"$\mu$ [1/mm]")

    im2 = ax2.imshow(error, cmap="RdBu_r", vmin=-lim, vmax=lim, extent=ext)
    ax2.set_title("Reconstruction error")
    fig.colorbar(im2, ax=ax2, label=r"$\Delta\mu$ [1/mm]")

    for ax in (ax1, ax2):
        ax.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    fig.tight_layout()
    finish(fig, f"{tag}_reconstruction")


def error_plots(sample, pixel_size, kVps=np.arange(40, 151, 10), filter_mm_Al=1.0, N0=1e4):
    """Reconstruction error against tube voltage for a sample from hydrate_growth. N0=None means no noise."""
    hyd, wat = sample.labels == LABEL["hydrate"], sample.labels == LABEL["water"]
    rms_all, ratio, E_effs = [], [], []
    for kVp in kVps:
        E, S = energy_spectrum(kVp=kVp, filter_mm_Al=filter_mm_Al)
        E_eff = np.sum(E * S) / np.sum(S)                  # mean energy of this spectrum
        img = sample.mu_map(E_eff)                         # reference image
        _, recon, _, _ = simulation(sample, E, S, pixel_size, N0=N0)
        err = recon - img
        contrast = img[hyd].mean() - img[wat].mean()
        rms_all.append(np.sqrt(np.mean(err**2)))
        ratio.append(np.sqrt(np.mean(err[sample.pore]**2)) / contrast)
        E_effs.append(E_eff)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.semilogy(kVps, rms_all, "o-")
    ax1.set_xlabel("Tube voltage [kVp]")
    ax1.set_ylabel("RMS error [1/mm]")
    ax1.set_title("Reconstruction error (whole image)")

    ax2.plot(kVps, ratio, "o-")
    ax2.axhline(1, color="grey", ls="--", lw=1)
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel("Tube voltage [kVp]")
    ax2.set_ylabel("RMS error in pores /\nhydrate–water contrast")
    ax2.set_title("Error relative to the hydrate contrast\n(below the dashed line hydrate stands out)")

    # Mean energy on a second x-axis at the top
    top = ax2.secondary_xaxis("top", functions=(lambda x: np.interp(x, kVps, E_effs),
                                                lambda e: np.interp(e, E_effs, kVps)))
    top.set_xlabel("Mean energy [keV]")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    ps, n, r = 0.02, 300, 2.6

    simple = Sample2D(n_pixel=n, pixel_size=ps, radius=r, seed=1)
    simple.simple_sample([("bone", 0.5), ("iron", 0.3), ("water", 0.5)], container="aluminum")

    rock = Sample2D(n_pixel=n, pixel_size=ps, radius=r, seed=1)
    rock.rock_sample(porosity=0.23, grain_radius=0.12)
    stages = rock.growth_stages(S_H=0.3, S_CO2=0.2)        # ends with rock.labels = final stage

    E, S = energy_spectrum(kVp=100, filter_mm_Al=1.0)
    E_eff = np.sum(E * S) / np.sum(S)                      # line 1: once, right after the spectrum

    for tag, sample in (("simple", simple), ("rock", rock)):
        img = sample.mu_map(E_eff)                         # line 2: inside the loop
        proj, recon, I, angles = simulation(sample, E, S, ps)    # line 3
        simu_plots(img, proj, recon, I, n, ps, angles, tag=tag)           # line 4

    error_plots(rock, ps, N0=1e4)
