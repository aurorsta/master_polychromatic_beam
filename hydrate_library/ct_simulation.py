"""
Simulated dynamic micro-CT scan of the hydrate phantom, written in the input format of NeCT.

    python simulate_scan.py

writes a folder scans/<date>_<time>/ with
    projections.npy   float32 [n_projections, detector rows, detector columns], line integrals -ln(I/I0)
    geometry.yaml     scanner geometry, angles [rad] and acquisition time of every projection [s]
    truth_mu.npy      true attenuation volumes [n_truth, z, y, x] at the times in truth_times.npy (for comparison)
    truth_labels.npy  true material labels at the same times
    reconstruct.py    the NeCT call, to run where NeCT is installed (needs an NVIDIA GPU)

Every projection is taken at its own time, so the sample changes during the scan just as in a real 4D scan.
Angles follow the modified golden-angle scheme of Friis and Nese (2024): equal steps within a revolution, and
each revolution starts shifted by a golden-ratio fraction of one step.

Geometry convention (as in TIGRE, which NeCT follows): the sample rotates about z; at angle a the source is at
DSO (cos a, sin a, 0), the detector centre at -(DSD - DSO) (cos a, sin a, 0), detector columns along
(-sin a, cos a, 0) and rows along z. Check this convention once with a static scan before trusting the
dynamic result (see the note at the end of this file).
"""
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from hydrate_phantom import CoreSample, HydrateGrowth

GOLDEN = (1 + np.sqrt(5)) / 2


# ---------------------------------------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------------------------------------
def golden_angle_schedule(n_per_rev, n_rev, t_start, t_end):
    """Angles [rad] and acquisition times [s] for n_rev revolutions of n_per_rev projections, evenly spread in time."""
    step = 2 * np.pi / n_per_rev
    offsets = (np.arange(n_rev) * step / GOLDEN) % step                 # start of each revolution
    angles = (offsets[:, None] + np.arange(n_per_rev)[None, :] * step).ravel()
    times = np.linspace(t_start, t_end, angles.size)
    return angles, times


# ---------------------------------------------------------------------------------------------------------
# Cone-beam forward projector (CPU, ray marching with trilinear interpolation)
# ---------------------------------------------------------------------------------------------------------
class ConeBeam:
    """
    Line integrals of a voxel volume for a circular cone-beam scan.

    DSO, DSD   : source-to-rotation-axis and source-to-detector distances [mm]
    nDetector  : [rows, columns];  dDetector: pixel size [rows, columns] [mm]
    nVoxel     : [z, y, x];        dVoxel: voxel size [mm] (cubic voxels)
    step       : ray marching step as a fraction of a voxel (0.5 = two samples per voxel)
    """

    def __init__(self, DSO, DSD, nDetector, dDetector, nVoxel, dVoxel, step=0.5):
        self.DSO, self.DSD = DSO, DSD
        self.nDetector, self.dDetector = list(nDetector), list(dDetector)
        self.nVoxel, self.dVoxel = list(nVoxel), float(dVoxel)
        self.ds = step * self.dVoxel
        nv, nu = self.nDetector
        self.v = (np.arange(nv) - (nv - 1) / 2) * dDetector[0]          # detector row positions [mm]
        self.u = (np.arange(nu) - (nu - 1) / 2) * dDetector[1]          # detector column positions [mm]
        nz, ny, nx = self.nVoxel
        self.half = np.array([nz, ny, nx]) * self.dVoxel / 2             # half size of the volume [mm]
        self.R = np.hypot(self.half[1], self.half[2])                    # radius that encloses the volume

    def project(self, mu, angle):
        """Line integrals [n_rows, n_columns] of mu [1/mm] at one angle [rad]."""
        c, s = np.cos(angle), np.sin(angle)
        src = np.array([self.DSO * c, self.DSO * s, 0.0])                # (x, y, z)
        U, V = np.meshgrid(self.u, self.v)                               # [rows, columns]
        det = np.stack([-(self.DSD - self.DSO) * c - U * s,
                        -(self.DSD - self.DSO) * s + U * c,
                        V], axis=-1)
        d = det - src
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        # entry and exit of each ray through the cylinder of radius R around z (the volume is inside it)
        a = d[..., 0]**2 + d[..., 1]**2
        b = 2 * (src[0] * d[..., 0] + src[1] * d[..., 1])
        disc = np.maximum(b**2 - 4 * a * (src[0]**2 + src[1]**2 - self.R**2), 0)
        t0 = (-b - np.sqrt(disc)) / (2 * a)
        t1 = (-b + np.sqrt(disc)) / (2 * a)
        n = int(np.ceil(2 * self.R / self.ds)) + 1
        k = np.arange(n) * self.ds
        out = np.zeros(self.nDetector)
        rows = max(1, int(4e6 // (n * self.nDetector[1])))               # rays in chunks, to bound memory
        for r0 in range(0, self.nDetector[0], rows):
            sl = slice(r0, r0 + rows)
            t = t0[sl, :, None] + k                                      # [rows, cols, n] distance along ray
            inside = t <= t1[sl, :, None]
            p = src + t[..., None] * d[sl, :, None, :]                   # sample points (x, y, z) [mm]
            idx = np.stack([(p[..., 2] + self.half[0]) / self.dVoxel - 0.5,    # -> voxel index (z, y, x)
                            (p[..., 1] + self.half[1]) / self.dVoxel - 0.5,
                            (p[..., 0] + self.half[2]) / self.dVoxel - 0.5])
            val = ndi.map_coordinates(mu, idx.reshape(3, -1), order=1, mode="constant", cval=0.0)
            out[sl] = (val.reshape(t.shape) * inside).sum(-1) * self.ds
        return out


# ---------------------------------------------------------------------------------------------------------
# Writing the scan
# ---------------------------------------------------------------------------------------------------------
def write_yaml(path, geo):
    """Small YAML writer (no extra dependency): numbers, strings and lists of numbers."""
    def fmt(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return "[" + ", ".join(fmt(x) for x in np.asarray(v).tolist()) + "]"
        if isinstance(v, (bool, np.bool_)):
            return "true" if v else "false"
        if isinstance(v, str):
            return v
        return repr(float(v)) if isinstance(v, (float, np.floating)) else str(v)
    Path(path).write_text("".join(f"{k}: {fmt(v)}\n" for k, v in geo.items()))


RECONSTRUCT_PY = '''"""Dynamic NeCT reconstruction of the simulated scan in this folder. Run where NeCT and a GPU are available."""
from pathlib import Path
import numpy as np
import nect

here = Path(__file__).parent
g = np.load(here / "geometry.npz")
geometry = nect.Geometry(
    DSD=float(g["DSD"]), DSO=float(g["DSO"]),
    nDetector=g["nDetector"].tolist(), dDetector=g["dDetector"].tolist(),
    nVoxel=g["nVoxel"].tolist(), dVoxel=g["dVoxel"].tolist(),
    angles=g["angles"], timesteps=g["timesteps"],
    mode="cone", angles_type="radians",
)
path = nect.reconstruct(geometry=geometry, projections=str(here / "projections.npy"),
                        quality="high", mode="dynamic", model="mixedcubes")
print("reconstruction saved in", path)
'''


def simulate_scan(hyd, out_dir, energy=25.0, n_per_rev=50, n_rev=40, t_start=0.0, t_end=None,
                  DSO=20.0, DSD=200.0, n_det=128, I0=1e4, step=0.5, n_truth=8, seed=0):
    """
    Simulate a dynamic cone-beam scan of the phantom and save it in NeCT's input format.

    energy        : monochromatic X-ray energy [keV]; ~25 keV suits a 2 mm sandstone (about 50 % transmission)
    n_per_rev     : projections per revolution;  n_rev: number of revolutions
    t_start, t_end: scan start and end [s] (default: whole experiment, 0 to end of hydrate growth)
    DSO, DSD      : distances [mm]; the detector pixel is chosen so that it maps to one voxel at the axis
    n_det         : detector size in pixels (square)
    I0            : photons per detector pixel without sample, for Poisson noise (None = no noise)
    n_truth       : number of ground-truth volumes saved, evenly spread over the scan
    """
    core = hyd.core
    t_end = hyd.end if t_end is None else t_end
    h = core.voxel_size
    nVoxel = list(core.core.shape)
    dDet = h * DSD / DSO                                                # one voxel at the axis = one pixel
    proj = ConeBeam(DSO, DSD, [n_det, n_det], [dDet, dDet], nVoxel, h, step)

    # the detector must see the whole core from every angle
    need = 2 * core.radius * DSD / np.sqrt(DSO**2 - core.radius**2), (core.height / 2) * DSD / (DSO - core.radius) * 2
    if n_det * dDet < max(need):
        raise ValueError(f"detector too small: needs {max(need) / dDet:.0f} pixels, has {n_det}")

    angles, times = golden_angle_schedule(n_per_rev, n_rev, t_start, t_end)
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    projections = np.zeros((angles.size, n_det, n_det), np.float32)
    mu_cache, key_cache = None, None
    t0 = time.time()
    for i, (a, t) in enumerate(zip(angles, times)):
        labels = hyd.labels_at(t)
        key = hash(labels.tobytes())                                    # reuse mu while nothing changes
        if key != key_cache:
            mu_cache, key_cache = hyd.mu_map(t, energy), key
        p = proj.project(mu_cache, a)
        if I0 is not None:
            counts = np.maximum(rng.poisson(I0 * np.exp(-p)), 1)
            p = -np.log(counts / I0)
        projections[i] = p
        if i % 100 == 0:
            print(f"projection {i}/{angles.size}, t = {t:.0f} s  ({time.time() - t0:.0f} s)")

    truth_t = np.linspace(t_start, t_end, n_truth)
    np.save(out_dir / "projections.npy", projections)
    np.save(out_dir / "truth_times.npy", truth_t)
    np.save(out_dir / "truth_mu.npy", np.stack([hyd.mu_map(t, energy) for t in truth_t]).astype(np.float32))
    np.save(out_dir / "truth_labels.npy", np.stack([hyd.labels_at(t) for t in truth_t]))
    geo = dict(mode="cone", DSO=DSO, DSD=DSD, nDetector=[n_det, n_det], dDetector=[dDet, dDet],
               nVoxel=nVoxel, dVoxel=[h, h, h], radians=True, angles=angles, timesteps=times)
    write_yaml(out_dir / "geometry.yaml", geo)
    np.savez(out_dir / "geometry.npz", **{k: v for k, v in geo.items() if k not in ("mode", "radians")})
    (out_dir / "reconstruct.py").write_text(RECONSTRUCT_PY)
    (out_dir / "README.txt").write_text(
        "Simulated dynamic cone-beam micro-CT scan of a CO2 hydrate phantom (sandstone core), for NeCT.\n\n"
        f"projections.npy   float32 {list(projections.shape)} = [projections, detector rows, detector columns],\n"
        "                  line integrals -ln(I/I0) (mu in 1/mm times length in mm)\n"
        "geometry.yaml     geometry in mm, angles in radians, timesteps = acquisition time of each projection [s]\n"
        "geometry.npz      the same geometry, read by reconstruct.py\n"
        "reconstruct.py    nect.reconstruct(..., mode='dynamic'); run with: python reconstruct.py\n"
        "truth_mu.npy      true attenuation [1/mm], shape [time, z, y, x], at the times in truth_times.npy\n"
        "truth_labels.npy  true materials, 0 outside, 1 quartz, 2 feldspar, 3 kaolinite, 4 water, 5 CO2, 6 hydrate\n\n"
        f"Energy {energy} keV (monochromatic), I0 = {I0} photons/pixel (Poisson noise), "
        f"{n_per_rev} projections/revolution x {n_rev} revolutions, modified golden angle,\n"
        f"scan from {t_start:.0f} s to {t_end:.0f} s. Voxel {h} mm, volume {nVoxel} [z, y, x].\n"
        "Geometry follows the TIGRE convention; please check it with a static reconstruction first.\n")
    print(f"saved {angles.size} projections in {out_dir}  ({time.time() - t0:.0f} s)")
    import shutil
    zip_path = shutil.make_archive(str(out_dir), "zip", out_dir)
    print(f"zipped for sending: {zip_path}")
    return projections, angles, times


if __name__ == "__main__":
    core = CoreSample(height=2.0, radius=1.0, voxel_size=0.02, porosity=0.23, grain_radius=0.12)
    core.cylinder()
    core.make_sample()
    core.set_start_water_saturation(1.0)
    core.inject_co2(start=0, duration=300, pump_rate=0.1)
    hyd = HydrateGrowth(core, pressure=3.0, temperature=274.15, induction_time=120)
    hyd.grow(start=300, duration=3600)

    out = Path(__file__).parent / "scans" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    simulate_scan(hyd, out, energy=25, n_per_rev=50, n_rev=40)

# Checking the geometry convention:
# simulate a static scan (e.g. t_start = t_end = 0, one revolution of 400 projections, mode="static" in
# reconstruct.py) and compare the NeCT result with truth_mu[0]. If it is mirrored, rotated or blurred,
# the angle direction or the detector orientation differs from NeCT's: negate the angles or flip the
# projections along the column axis (projections[:, :, ::-1]) and try again.
