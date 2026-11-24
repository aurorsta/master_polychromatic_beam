from samples_2D import Sample2D, NAMES, LABEL, mu
from skimage.transform import radon, iradon, rescale 
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
# One folder per script and run: figures/parallell_monochromatic_sim/2026-11-24_1548/
FIG_DIR = (Path(__file__).resolve().parent / "figures" / Path(__file__).stem
           / datetime.now().strftime("%Y-%m-%d_%H%M"))


def finish(fig, name):
    """Save the figure to FIG_DIR without showing it."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# One colour per material label; anything not listed is black
PHASE_COLORS = {"vacuum": "white", "sandstone": "#b9975b", "water": "#3b7dd8",
                "CO2 gas": "#f0c419", "hydrate": "#f10808"}

def simulation(img, pixel_size, theta_max=180, N0=None, seed=0):
    """Parallel beam, monochromatic. N0 = photons per detector pixel; None means no noise."""
    theta = np.linspace(0.0, theta_max, max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)                 # sum of mu along each ray [1/mm * px]
    I = np.exp(-sinogram * pixel_size)                 # I/I0, Beer-Lambert
    if N0 is not None:
        rng = np.random.default_rng(seed)
        counts = np.maximum(rng.poisson(N0 * I), 1)    # Poisson noise on photon counts
        I = counts / N0
        sinogram = -np.log(I) / pixel_size
    recon = iradon(sinogram, theta=theta, filter_name="ramp")
    return sinogram, recon, I, theta


def simu_plots(img, sino, recon, I, n_pixel, pixel_size, theta, tag=""):
    L_img = n_pixel * pixel_size
    ext = (0, L_img, L_img, 0)                                     # image extent [mm]
    dtheta = theta[1] - theta[0]                                   # angle step [deg]
    det_len = sino.shape[0] * pixel_size                           # detector length [mm]
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
    im2 = ax2.imshow(sino * pixel_size, cmap="Greys_r", extent=ext_sino,
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
    finish(fig, f"{tag}detector_intensity")

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

def plot_stages(stages, pixel_size):
    """stages: list of (labels, title) from Sample2D.growth_stages."""
    n = stages[0][0].shape[0]
    L = n * pixel_size
    ext = (0, L, L, 0)
    cmap = ListedColormap([PHASE_COLORS.get(name, "black") for name in NAMES])

    fig, axes = plt.subplots(1, len(stages), figsize=(4.5 * len(stages), 5), sharex=True, sharey=True)
    for ax, (labels, title) in zip(axes, stages):
        ax.imshow(labels, cmap=cmap, vmin=-0.5, vmax=len(NAMES) - 0.5,
                  interpolation="nearest", extent=ext)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")

    # Legend with only the materials that appear
    present = np.unique(np.stack([labels for labels, _ in stages]))
    handles = [Patch(color=cmap(i), label=NAMES[i]) for i in present if NAMES[i] != "vacuum"]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    finish(fig, "hydrate_formation_stages")

def contrast_table(energies=(40, 60, 80, 100, 140),
                   names=("water", "ice", "hydrate", "CO2 gas", "quartz", "sandstone")):
    w = 12                                                     # column width
    header = f"{'material':{w}s}" + "".join(f"{f'{e} keV':>{w}s}" for e in energies)

    print("Linear attenuation coefficient mu [1/mm]")
    print(header)
    for name in names:
        print(f"{name:{w}s}" + "".join(f"{mu(name, e):{w}.5f}" for e in energies))

    print("\nContrast relative to water, (mu - mu_water) / mu_water [%]")
    print(header)
    for name in ("hydrate", "ice"):
        rel = [100 * (mu(name, e) - mu("water", e)) / mu("water", e) for e in energies]
        print(f"{name:{w}s}" + "".join(f"{r:+{w}.1f}" for r in rel))

    print("\nContrast relative to ice, (mu_hydrate - mu_ice) / mu_ice [%]")
    print(header)
    rel = [100 * (mu("hydrate", e) - mu("ice", e)) / mu("ice", e) for e in energies]
    print(f"{'hydrate':{w}s}" + "".join(f"{r:+{w}.1f}" for r in rel))


def error_plots(sample, pixel_size, energies=np.arange(20, 151, 10), N0=1e4):
    """Reconstruction error against energy for a sample from hydrate_growth. N0=None means no noise."""
    hyd, wat = sample.labels == LABEL["hydrate"], sample.labels == LABEL["water"]
    rms_all, ratio = [], []
    for E in energies:
        img = sample.mu_map(E)
        _, recon, _, _ = simulation(img, pixel_size, N0=N0)
        err = recon - img
        contrast = img[hyd].mean() - img[wat].mean()
        rms_all.append(np.sqrt(np.mean(err**2)))
        ratio.append(np.sqrt(np.mean(err[sample.pore]**2)) / contrast)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.semilogy(energies, rms_all, "o-")
    ax1.set_xlabel("Photon energy [keV]")
    ax1.set_ylabel("RMS error [1/mm]")
    ax1.set_title("Reconstruction error (whole image)")

    ax2.plot(energies, ratio, "o-")
    ax2.axhline(1, color="grey", ls="--", lw=1)
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel("Photon energy [keV]")
    ax2.set_ylabel("RMS error in pores /\nhydrate–water contrast")
    ax2.set_title("Error relative to the hydrate contrast\n(below the dashed line hydrate stands out)")
    fig.tight_layout()
    finish(fig, "error_plot")

if __name__ == "__main__":
    ps, n, r, energy = 0.02, 300, 2.6, 60

    simple = Sample2D(n_pixel=n, pixel_size=ps, radius=r, seed=1)
    simple.simple_sample([("bone", 0.5), ("iron", 0.3), ("water", 0.5)], container="aluminum")
    simple_img = simple.mu_map(energy)

    rock = Sample2D(n_pixel=n, pixel_size=ps, radius=r, seed=1)
    rock.rock_sample(porosity=0.23, grain_radius=0.12)
    rock.hydrate_growth(S_H=0.3, S_CO2=0.2)
    stages = rock.growth_stages(S_H=0.3, S_CO2=0.2)
    plot_stages(stages, ps)
    hydrate_img = rock.mu_map(energy)
    contrast_table()
    


    for tag, sample in (("simple", simple), ("rock", rock)):
        img = sample.mu_map(energy)
        sino, recon, I, theta = simulation(img, ps, N0=None)
        simu_plots(img, sino, recon, I, n, ps, theta, tag=tag)

    error_plots(rock, ps, N0=None)    