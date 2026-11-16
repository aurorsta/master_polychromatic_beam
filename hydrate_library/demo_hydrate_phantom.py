"""Example use of hydrate_phantom.py: five morphologies, one noisy scan, and the medical-CT estimate."""
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import hydrate_phantom as hp

n_pixel, pixel_size = 512, 0.01                 # 10 um pixels -> 5.12 mm image
core_radius, sleeve = 2.2, 0.3                  # mm
theta = np.linspace(0, 180, 360, endpoint=False)
E = [60.0]; w = [1.0]                           # monochromatic 60 keV (swap in a spectrum later)
Sw0, conv, sal0 = 0.5, 0.8, 0.035
morphs = ["coating", "filling", "cementing", "porous", "patchy"]

# ---- 1. Attenuation check: does salt hide the hydrate?
mus = hp.material_mus(60, salinity=sal0)
mus_end = hp.material_mus(60, salinity=min(sal0 / (1 - conv), 0.26))
print(f"mu at 60 keV [1/mm]: hydrate {mus[hp.HYDRATE]:.4f}, "
      f"brine {sal0*100:.1f} wt% {mus[hp.BRINE]:.4f}, "
      f"brine after conv={conv} ({sal0/(1-conv)*100:.1f} wt%) {mus_end[hp.BRINE]:.4f}")

# ---- 2. Five morphologies, same Sw0 and conv
cmap = ListedColormap(["white", "tan", "tab:blue", "tab:red", "lightgrey", "dimgrey"])
fig, axes = plt.subplots(1, 5, figsize=(20, 4.4))
ext = (0, n_pixel * pixel_size, n_pixel * pixel_size, 0)
samples = {}
for ax, m in zip(axes, morphs):
    t = time.time()
    labels, info = hp.hydrate_core(n_pixel, pixel_size, core_radius, sleeve, Sw0=Sw0, conv=conv,
                                   salinity0=sal0, morphology=m, seed=1)
    samples[m] = (labels, info)
    print(f"{m:10s} S_H={info['S_H']:.2f} S_w={info['S_w']:.2f} S_CO2={info['S_co2']:.2f} "
          f"salinity={info['salinity']*100:.1f} wt%  ({time.time()-t:.1f} s)")
    ax.imshow(labels, cmap=cmap, vmin=-0.5, vmax=5.5, interpolation="nearest", extent=ext)
    ax.set_xlim(1.6, 3.4); ax.set_ylim(3.4, 1.6)       # zoom on the centre
    ax.set_title(m)
    ax.set_xlabel("x [mm]")
axes[0].set_ylabel("y [mm]")
handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in (1, 2, 3, 4)]
fig.legend(handles, ["quartz", "brine", "hydrate", "CO$_2$"], loc="lower center", ncol=4)
fig.suptitle(f"Hydrate morphologies, $S_{{w0}}$={Sw0}, conv={conv}  (zoomed 1.8 mm window)")
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig("morphologies.png", dpi=130)

# ---- 3. Noisy scan of one sample, plus dry and wet reference scans (as in the medical CT method)
def reference(labels, fill):
    ref = labels.copy()
    pore = np.isin(labels, [hp.BRINE, hp.HYDRATE, hp.CO2])
    ref[pore] = fill
    return ref

rows = []
for m in morphs:
    labels, info = samples[m]
    kw = dict(salinity=info["salinity"])
    rec_exp = hp.reconstruct(hp.scan(labels, pixel_size, theta, E, w, N0=1e5, mus_kwargs=kw, rng=1), theta)
    rec_dry = hp.reconstruct(hp.scan(reference(labels, hp.CO2), pixel_size, theta, E, w, N0=1e5,
                                     mus_kwargs=dict(salinity=sal0, rho_co2=0.0012), rng=2), theta)
    rec_wet = hp.reconstruct(hp.scan(reference(labels, hp.BRINE), pixel_size, theta, E, w, N0=1e5,
                                     mus_kwargs=dict(salinity=sal0), rng=3), theta)

    # medical CT: 0.3 mm voxels (30 x 30 pixels), only voxels fully inside the core
    b = 30
    core_frac = hp.block_average(info["core"].astype(float), b)
    inner = core_frac == 1
    E_, D_, W_ = (hp.block_average(r, b) for r in (rec_exp, rec_dry, rec_wet))
    S_est = hp.medical_ct_saturation(E_, D_, W_)
    # core average: ratio of the averages (averaging the voxel ratios is biased by voxels with little pore space)
    S_core = (E_[inner] - D_[inner]).mean() / (W_[inner] - D_[inner]).mean()
    rows.append((m, info["S_w"] + info["S_H"], S_core, np.median(S_est[inner])))
    if m == "filling":
        keep = (labels, rec_exp, S_est, inner)

print("\nMedical-CT estimate of S_{w+H} vs truth")
print(f"{'morphology':10s} {'true':>6s} {'core est':>9s} {'voxel median':>13s}")
for m, true, est, med in rows:
    print(f"{m:10s} {true:6.2f} {est:9.2f} {med:13.2f}")

labels, rec_exp, S_est, inner = keep
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
axes[0].imshow(labels, cmap=cmap, vmin=-0.5, vmax=5.5, interpolation="nearest", extent=ext)
axes[0].set_title("True labels (filling)")
im = axes[1].imshow(rec_exp, cmap="Greys_r", vmin=0.9 * mus[hp.BRINE], vmax=1.15 * mus[hp.HYDRATE], extent=ext)
axes[1].set_title("Micro-CT reconstruction, 10 µm\nnarrow window (brine vs hydrate)")
fig.colorbar(im, ax=axes[1], label=r"$\mu$ [1/mm]", shrink=0.8)
S_show = np.where(inner, S_est, np.nan)
im = axes[2].imshow(S_show, cmap="viridis", vmin=0, vmax=1, extent=ext, interpolation="nearest")
axes[2].set_title("Medical-CT voxels, 0.3 mm\n$S_{w+H}$ from CT$_{wet}$/CT$_{dry}$")
fig.colorbar(im, ax=axes[2], label="$S_{w+H}$", shrink=0.8)
for ax in axes:
    ax.set_xlabel("x [mm]")
fig.tight_layout()
fig.savefig("scan_example.png", dpi=130)
