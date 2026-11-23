"""
Compare a dynamic NeCT reconstruction with the ground truth of the simulated scan. Needs a GPU (run on IDUN).

    python compare_dynamic.py <scan folder> <NeCT output folder>
    python compare_dynamic.py scans/2026-11-23_173036 outputs/2026-11-23_173036

The NeCT output folder can be any folder above the model; the script finds config.yaml and checkpoints/last.ckpt.
Writes into the scan folder:
    recon_mu.npy      reconstructed volumes at the truth times [time, z, y, x]
    comparison.png    middle XZ slice: truth, reconstruction and change since t = 0, at every truth time
and prints, for every truth time, how well the reconstruction matches the truth.
Add --video to also make NeCT's own video of the reconstruction.
"""
import argparse
from pathlib import Path

import numpy as np


def find_model_dir(path):
    "Folder that contains config.yaml and checkpoints/last.ckpt, at or below path"
    path = Path(path)
    for d in [path, *sorted(p.parent for p in path.rglob("config.yaml"))]:
        if (d / "config.yaml").exists() and (d / "checkpoints" / "last.ckpt").exists():
            return d
    raise FileNotFoundError(f"no folder with config.yaml and checkpoints/last.ckpt under {path}")


def orientations(vol):
    "All flips and 90 degree rotations in the xy plane, plus a flip in z"
    for flip_z in (False, True):
        for k in range(4):
            for flip_x in (False, True):
                r = vol[::-1] if flip_z else vol
                r = np.rot90(r, k, axes=(1, 2))
                r = r[:, :, ::-1] if flip_x else r
                name = f"rot90 x{k}" + (", flip x" if flip_x else "") + (", flip z" if flip_z else "")
                yield name, (flip_z, k, flip_x)


def apply(vol, how):
    flip_z, k, flip_x = how
    r = vol[::-1] if flip_z else vol
    r = np.rot90(r, k, axes=(1, 2))
    return np.ascontiguousarray(r[:, :, ::-1] if flip_x else r)


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("nect_output")
    ap.add_argument("--video", action="store_true", help="also export NeCT's video of the reconstruction")
    args = ap.parse_args()

    import nect
    import tifffile

    scan = Path(args.scan)
    model_dir = find_model_dir(args.nect_output)
    print("model:", model_dir)
    truth = np.load(scan / "truth_mu.npy")                       # [time, z, y, x] in 1/mm
    truth_t = np.load(scan / "truth_times.npy")                  # [s]
    times = np.load(scan / "geometry.npz")["timesteps"]          # acquisition time of every projection [s]

    # NeCT reads the projections again when exporting; point it to them with a full path
    import re
    cfg_file = model_dir / "config.yaml"
    proj_path = str((scan / "projections.npy").resolve())
    text = cfg_file.read_text()
    text = re.sub(r"(?m)^img_path:.*$", f"img_path: {proj_path}", text)
    cfg_file.write_text(text)

    # NeCT divides the timesteps by their maximum, so the truth times are converted the same way
    t_norm = [float(t / times.max()) for t in truth_t]
    vol_dir = nect.dynamic_export.export_volumes_at_timesteps(model_dir, t_norm)
    recon = np.stack([tifffile.imread(vol_dir / f"T{t:04}.tiff") for t in t_norm])

    # the exported volumes may be turned relative to the truth; pick the orientation that fits best at t = 0
    best = max(orientations(recon[0]), key=lambda o: corr(apply(recon[0], o[1]), truth[0])
               if apply(recon[0], o[1]).shape == truth[0].shape else -1)
    print("orientation:", best[0])
    recon = np.stack([apply(v, best[1]) for v in recon])

    # NeCT's values are in its own scale; a straight-line fit at t = 0 maps them to 1/mm
    a, b = np.polyfit(recon[0].ravel(), truth[0].ravel(), 1)
    recon = (a * recon + b).astype(np.float32)
    np.save(scan / "recon_mu.npy", recon)

    print("\n  t [s]   match   change match   (1 = perfect)")
    for i, t in enumerate(truth_t):
        whole = corr(recon[i], truth[i])
        if i == 0:
            print(f"  {t:5.0f}   {whole:.3f}")
            continue
        change = corr(recon[i] - recon[0], truth[i] - truth[0])        # does it see what moved?
        print(f"  {t:5.0f}   {whole:.3f}   {change:.3f}")
    print("'match' is dominated by the rock, which does not change. 'change match' compares only what\n"
          "changed since t = 0 (gas front and hydrate), so it is the number that tells if NeCT follows the dynamics.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        y = truth.shape[2] // 2
        n = len(truth_t)
        fig, ax = plt.subplots(4, n, figsize=(2.2 * n, 9), squeeze=False)
        lo, hi = truth.min(), truth.max()
        dlim = np.abs(truth - truth[0]).max()
        rows = [("truth", truth, "gray", lo, hi), ("NeCT", recon, "gray", lo, hi),
                ("truth change", truth - truth[0], "RdBu_r", -dlim, dlim),
                ("NeCT change", recon - recon[0], "RdBu_r", -dlim, dlim)]
        for r, (label, vols, cmap, vmin, vmax) in enumerate(rows):
            for i in range(n):
                ax[r, i].imshow(vols[i][:, y, :], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
                ax[r, i].set_xticks([]), ax[r, i].set_yticks([])
                if r == 0:
                    ax[r, i].set_title(f"t = {truth_t[i]:.0f} s")
            ax[r, 0].set_ylabel(label)
        fig.tight_layout()
        fig.savefig(scan / "comparison.png", dpi=120)
        print("saved", scan / "comparison.png")
    except ImportError:
        print("matplotlib not installed, no figure")

    if args.video:
        print("video:", nect.export_video(model_dir, plot_slice="XZ"))
