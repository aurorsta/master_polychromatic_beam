"""
GPU version of the simulated dynamic micro-CT scan, followed by the NeCT reconstruction. Made for IDUN.

    python scan_gpu.py config.yaml               dynamic scan, then dynamic NeCT reconstruction
    python scan_gpu.py config.yaml --static      static scan at t = 0 and static reconstruction (geometry check)
    python scan_gpu.py config.yaml --no-recon    only the projections

The phantom (Hydrate_core) runs on the CPU as before; the projections and the reconstruction run on the GPU.
Needs Hydrate_core.py and ct_simulation.py in the same folder.

Settings come from two optional sections of the config:
    scan:        energy, n_per_rev, n_rev, t_start, t_end, DSO, DSD, n_det, I0, step, n_truth, seed
                 (same names as in ct_simulation.py; n_det can be left out and is then sized to fit the core)
    reconstruct: quality, enc_arc, and any other keyword of nect.reconstruct (niter, lr, ...)

Output: scans/<date>_<time>/ with projections.npy, geometry.yaml/.npz, truth_mu.npy, truth_labels.npy,
truth_times.npy. NeCT writes its result to outputs/<same name>/ in the folder you run from.
"""
import argparse
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import Hydrate_core as hc
from ct_simulation import build_phantom, golden_angle_schedule, write_yaml


# ---------------------------------------------------------------------------------------------------------
# Cone-beam forward projector on the GPU (same geometry and math as ConeBeam in ct_simulation.py)
# ---------------------------------------------------------------------------------------------------------
class ConeBeamTorch:
    """
    Line integrals of a voxel volume for a circular cone-beam scan, TIGRE convention.

    DSO, DSD   : source-to-rotation-axis and source-to-detector distances [mm]
    nDetector  : [rows, columns];  dDetector: pixel size [rows, columns] [mm]
    nVoxel     : [z, y, x];        dVoxel: voxel size [mm] (cubic voxels)
    step       : ray marching step as a fraction of a voxel (0.5 = two samples per voxel)
    max_points : sample points per chunk; lower it if the GPU runs out of memory
    """

    def __init__(self, DSO, DSD, nDetector, dDetector, nVoxel, dVoxel, step=0.5, device="cuda",
                 max_points=2**24):
        self.DSO, self.DSD = float(DSO), float(DSD)
        self.nDetector = [int(n) for n in nDetector]
        self.dVoxel = float(dVoxel)
        self.ds = step * self.dVoxel
        self.device = torch.device(device)
        self.max_points = int(max_points)
        nv, nu = self.nDetector
        v = (torch.arange(nv, dtype=torch.float32, device=self.device) - (nv - 1) / 2) * dDetector[0]
        u = (torch.arange(nu, dtype=torch.float32, device=self.device) - (nu - 1) / 2) * dDetector[1]
        self.V, self.U = torch.meshgrid(v, u, indexing="ij")            # [rows, columns]
        nz, ny, nx = nVoxel
        self.half = [nz * self.dVoxel / 2, ny * self.dVoxel / 2, nx * self.dVoxel / 2]   # [mm] (z, y, x)
        self.R = math.hypot(self.half[1], self.half[2])                 # radius that encloses the volume
        self.n = int(math.ceil(2 * self.R / self.ds)) + 1               # samples per ray
        self.k = torch.arange(self.n, dtype=torch.float32, device=self.device) * self.ds
        # grid_sample wants (x, y, z) in [-1, 1]; with align_corners=False that is position / half size
        self.scale = torch.tensor([1 / self.half[2], 1 / self.half[1], 1 / self.half[0]], device=self.device)

    @torch.no_grad()
    def project(self, vol, angle):
        """Line integrals [rows, columns] of vol [1/mm], a tensor of shape (1, 1, z, y, x) on the device."""
        c, s = math.cos(angle), math.sin(angle)
        src = torch.tensor([self.DSO * c, self.DSO * s, 0.0], device=self.device)
        det = torch.stack([-(self.DSD - self.DSO) * c - self.U * s,
                           -(self.DSD - self.DSO) * s + self.U * c,
                           self.V], dim=-1)
        d = det - src
        d = d / d.norm(dim=-1, keepdim=True)
        # entry and exit of each ray through the cylinder of radius R around z
        a = d[..., 0] ** 2 + d[..., 1] ** 2
        b = 2 * (src[0] * d[..., 0] + src[1] * d[..., 1])
        disc = (b ** 2 - 4 * a * (src[0] ** 2 + src[1] ** 2 - self.R ** 2)).clamp(min=0)
        t0 = (-b - disc.sqrt()) / (2 * a)
        t1 = (-b + disc.sqrt()) / (2 * a)

        out = torch.empty(self.nDetector, device=self.device)
        rows = max(1, self.max_points // (self.n * self.nDetector[1]))
        for r0 in range(0, self.nDetector[0], rows):
            sl = slice(r0, r0 + rows)
            t = t0[sl, :, None] + self.k                                # [rows, cols, n]
            inside = t <= t1[sl, :, None]
            p = src + t[..., None] * d[sl, :, None, :]                  # (x, y, z) [mm]
            grid = (p * self.scale).reshape(1, 1, 1, -1, 3)
            val = F.grid_sample(vol, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            out[sl] = (val.reshape(t.shape) * inside).sum(-1) * self.ds
        return out


# ---------------------------------------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------------------------------------
def detector_size(radius, height, h, DSO, DSD, n_det=None):
    "Detector [rows, columns] that sees the whole core from every angle, with a small margin"
    dDet = h * DSD / DSO                                                # one voxel at the axis = one pixel
    need_cols = math.ceil(2 * radius * DSD / math.sqrt(DSO ** 2 - radius ** 2) / dDet)
    need_rows = math.ceil(height * DSD / (DSO - radius) / dDet)
    if n_det is None:
        return [need_rows + 8, need_cols + 8], dDet
    n_det = [n_det, n_det] if np.isscalar(n_det) else list(n_det)
    if n_det[0] < need_rows or n_det[1] < need_cols:
        raise ValueError(f"detector too small: needs [{need_rows}, {need_cols}] pixels, has {n_det}")
    return n_det, dDet


def simulate_scan_gpu(tl, out_dir, energy=25.0, n_per_rev=50, n_rev=40, t_start=0.0, t_end=None,
                      DSO=20.0, DSD=200.0, n_det=None, I0=1e4, step=0.5, n_truth=6, seed=0,
                      device="cuda", max_points=2**24):
    """Simulate a dynamic cone-beam scan of the phantom on the GPU and save it in NeCT's input format."""
    core = tl.sample
    t_end = tl.end if t_end is None else t_end
    h = core.voxel_size_um * 1e-3                                       # voxel size [mm]
    nVoxel = list(core.shape)                                           # [z, y, x]
    nDet, dDet = detector_size(core.radius_mm, core.height_mm, h, DSO, DSD, n_det)
    proj = ConeBeamTorch(DSO, DSD, nDet, [dDet, dDet], nVoxel, h, step, device, max_points)
    tl.set_energy(energy)

    angles, times = golden_angle_schedule(n_per_rev, n_rev, t_start, t_end)
    truth_t = np.linspace(t_start, t_end, n_truth)
    gen = torch.Generator(device=device).manual_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"volume {nVoxel}, detector {nDet}, {proj.n} samples per ray, {angles.size} projections")

    projections = np.zeros((angles.size, *nDet), np.float32)
    truth_mu, truth_labels, j = [], [], 0
    vol, k_cached = None, None
    t0 = time.time()
    for i, (a, t) in enumerate(zip(angles, times)):
        # ground truth is collected on the way, since the simulation only runs forward
        while j < n_truth and truth_t[j] <= t:
            tl.advance_to(truth_t[j])
            truth_mu.append(tl.mu())
            truth_labels.append(tl.labels())
            j += 1
        k = tl.advance_to(t)
        if k != k_cached:                                               # new volume only when the step changes
            vol = torch.from_numpy(tl.mu()).to(device)[None, None]
            k_cached = k
        p = proj.project(vol, a)
        if I0 is not None:
            counts = torch.poisson(I0 * torch.exp(-p), generator=gen).clamp_(min=1)
            p = -torch.log(counts / I0)
        projections[i] = p.cpu().numpy()
        if i % 100 == 0:
            print(f"projection {i}/{angles.size}, t = {t:.0f} s  ({time.time() - t0:.0f} s)")
    while j < n_truth:
        tl.advance_to(truth_t[j])
        truth_mu.append(tl.mu())
        truth_labels.append(tl.labels())
        j += 1
    del vol
    if torch.cuda.is_available():
        torch.cuda.empty_cache()                                        # leave the GPU memory to NeCT

    np.save(out_dir / "projections.npy", projections)
    np.save(out_dir / "truth_times.npy", truth_t)
    np.save(out_dir / "truth_mu.npy", np.stack(truth_mu))
    np.save(out_dir / "truth_labels.npy", np.stack(truth_labels))
    geo = dict(mode="cone", DSO=float(DSO), DSD=float(DSD), nDetector=nDet, dDetector=[dDet, dDet],
               nVoxel=nVoxel, dVoxel=[h, h, h], radians=True, angles=angles, timesteps=times)
    write_yaml(out_dir / "geometry.yaml", geo)
    np.savez(out_dir / "geometry.npz", **{k: v for k, v in geo.items() if k not in ("mode", "radians")})
    labels = ", ".join(f"{i} {name}" for i, name in enumerate(hc.PHASES))
    (out_dir / "README.txt").write_text(
        f"Simulated cone-beam micro-CT scan of a CO2 hydrate phantom. Energy {energy} keV, I0 = {I0}, "
        f"{n_per_rev} projections/revolution x {n_rev} revolutions, {t_start:.0f} to {t_end:.0f} s.\n"
        f"projections.npy [projections, rows, columns], line integrals. truth_mu.npy [time, z, y, x] in 1/mm.\n"
        f"truth_labels.npy dominant phase: {labels}\n")
    print(f"saved {angles.size} projections in {out_dir}  ({time.time() - t0:.0f} s)")
    return geo


# ---------------------------------------------------------------------------------------------------------
# Reconstruction with NeCT
# ---------------------------------------------------------------------------------------------------------
def reconstruct(out_dir, geo, mode="dynamic", quality="high", enc_arc="mixedcubes", **kwargs):
    "Run nect.reconstruct on the saved scan. Returns what NeCT returns."
    import nect                                                         # only needed here
    geometry = nect.Geometry(
        DSD=geo["DSD"], DSO=geo["DSO"],
        nDetector=geo["nDetector"], dDetector=geo["dDetector"],
        nVoxel=geo["nVoxel"], dVoxel=geo["dVoxel"],
        angles=geo["angles"], radians=True,
        timesteps=geo["timesteps"] if mode == "dynamic" else None,
        mode="cone",
    )
    return nect.reconstruct(geometry=geometry, projections=str(Path(out_dir) / "projections.npy"),
                            mode=mode, quality=quality, enc_arc=enc_arc if mode == "dynamic" else None,
                            exp_name=Path(out_dir).name, **kwargs)


def check_orientation(recon, truth):
    "Correlation of the reconstruction with the truth for every flip and 90 degree rotation; best first"
    scores = []
    for flip_z in (False, True):
        for k in range(4):
            for flip_x in (False, True):
                r = recon[::-1] if flip_z else recon
                r = np.rot90(r, k, axes=(1, 2))
                r = r[:, :, ::-1] if flip_x else r
                if r.shape != truth.shape:
                    continue
                name = f"rot90 x{k}" + (", flip x" if flip_x else "") + (", flip z" if flip_z else "")
                scores.append((float(np.corrcoef(r.ravel(), truth.ravel())[0, 1]), name))
    return sorted(scores, reverse=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="config.yaml")
    ap.add_argument("--static", action="store_true", help="static scan at t = 0 to check the geometry")
    ap.add_argument("--static-projections", type=int, default=400)
    ap.add_argument("--no-recon", action="store_true", help="only make the projections")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    scan = dict(cfg.get("scan") or {})
    rec = dict(cfg.get("reconstruct") or {})
    if args.static:
        scan.update(t_start=0.0, t_end=0.0, n_rev=1, n_per_rev=args.static_projections, n_truth=1)
        rec["mode"] = "static"

    tl = build_phantom(cfg)
    name = datetime.now().strftime("%Y-%m-%d_%H%M%S") + ("_static" if args.static else "")
    out = Path("scans") / name
    geo = simulate_scan_gpu(tl, out, device=args.device, **scan)

    if not args.no_recon:
        result = reconstruct(out, geo, **rec)
        if args.static:
            volume = np.asarray(result[0])
            np.save(out / "recon_static.npy", volume)
            truth = np.load(out / "truth_mu.npy")[0]
            print("orientation check: correlation with the truth after turning the reconstruction, best first")
            for score, how in check_orientation(volume, truth)[:4]:
                print(f"  {score:.3f}  {how}")
            print("If the best one is 'rot90 x0' the geometry matches NeCT.")
        else:
            print("reconstruction saved in", result[0])
