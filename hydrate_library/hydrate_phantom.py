"""
3D phantom of a sandstone core where CO2 is injected and CO2 hydrate then grows, for CT simulation.

The two classes are used one after the other, in the order of the experiment:

    core = CoreSample(height=2.0, radius=1.0, porosity=0.23, grain_radius=0.12)
    core.cylinder()                               # the empty cylinder
    core.make_sample()                            # grains and pore space inside it
    core.set_start_water_saturation(1.0)          # pores fully water saturated at t = 0
    core.inject_co2(start=0, duration=300)        # CO2 pumped in from the bottom for 300 s

    hyd = HydrateGrowth(core, pressure=3.0, temperature=274.15)
    hyd.grow(start=300, duration=3600)            # hydrate conditions from t = 300 s, for one hour

    labels = hyd.labels_at(t)                     # material of every voxel at time t
    mu_vox = hyd.mu_map(t, energy=100)            # linear attenuation of every voxel at time t

Every voxel stores the time at which CO2 and hydrate reach it, so the state at any time is a cheap lookup.
Arrays are [z, y, x]: z is the core axis (and the scanner's rotation axis), CO2 enters at z = 0.
Units: mm, 1/mm, keV, s, MPa, K.
"""
import numpy as np
from scipy import ndimage as ndi
import xraylib as xrl

# ---------------------------------------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------------------------------------
# name -> (chemical formula, density [g/cm^3])
MATERIALS = {
    "water":     ("H2O",           1.000),
    "ice":       ("H2O",           0.917),
    "hydrate":   ("CO2(H2O)6.08",  1.09),
    "CO2 gas":   ("CO2",           0.07),
    "quartz":    ("SiO2",          2.65),
    "feldspar":  ("KAlSi3O8",      2.56),
    "kaolinite": ("Al2Si2O5(OH)4", 2.60),
}

# Bentheimer grain composition, XRD weight fractions (Peksa et al. 2015, Table 1, "Bentheimer sandstone revisited for experimental purposes").
# Clay = kaolinite 2.50 + montmorillonite 0.18; the remaining 0.7 wt% (carbonates, oxides, pyrite) is left out.
SANDSTONE_WT = {"quartz": 0.9170, "feldspar": 0.0486, "kaolinite": 0.0268}
_vol = {m: w / MATERIALS[m][1] for m, w in SANDSTONE_WT.items()}   # weight / density = volume
SANDSTONE = {m: v / sum(_vol.values()) for m, v in _vol.items()}  # volume fractions, sum to 1

# Integer label of every voxel. 0 is outside the core.
OUTSIDE, QUARTZ, FELDSPAR, KAOLINITE, WATER, CO2, HYDRATE = range(7)
LABEL_MATERIAL = {QUARTZ: "quartz", FELDSPAR: "feldspar", KAOLINITE: "kaolinite",
                  WATER: "water", CO2: "CO2 gas", HYDRATE: "hydrate"}
MINERAL_LABEL = {name: lab for lab, name in LABEL_MATERIAL.items() if lab in (QUARTZ, FELDSPAR, KAOLINITE)}

# Hydrate volume / volume of the water it consumed, from the hydration number and density (~1.29)
N_H, RHO_H = 6.08, 1.09
M_H2O, M_CO2 = 18.015, 44.01                                     # molar masses [g/mol]
EXPANSION = ((M_H2O + M_CO2 / N_H) / RHO_H) / M_H2O


def mu(name, energy):
    """Linear attenuation [1/mm] of a material at one or several energies [keV]."""
    if name == "sandstone":
        return sum(f * mu(m, energy) for m, f in SANDSTONE.items())  # volume-weighted average of the minerals
    formula, rho = MATERIALS[name]
    E = np.atleast_1d(np.asarray(energy, float))
    out = np.array([xrl.CS_Total_CP(formula, float(e)) * rho / 10 for e in E])  # [cm^2/g]*[g/cm^3] = [1/cm], /10 -> [1/mm]
    return out if np.ndim(energy) else out[0]


def mu_map(labels, energy):
    """Linear attenuation [1/mm] of every voxel in a label array, at one energy [keV]."""
    table = np.zeros(max(LABEL_MATERIAL) + 1)
    for lab, name in LABEL_MATERIAL.items():
        table[lab] = mu(name, energy)
    return table[labels]


def saturations(labels):
    """Fraction of the pore volume filled with water, CO2 and hydrate."""
    n_pore = np.isin(labels, (WATER, CO2, HYDRATE)).sum()
    return {name: (labels == lab).sum() / n_pore for lab, name in ((WATER, "water"), (CO2, "CO2"), (HYDRATE, "hydrate"))}


# ---------------------------------------------------------------------------------------------------------
# Core sample: rock, start saturation, CO2 injection
# ---------------------------------------------------------------------------------------------------------
class CoreSample:
    """
    A cylindrical sandstone core.

    height, radius : size of the cylinder [mm]
    voxel_size     : edge length of one voxel [mm]
    porosity       : pore volume / core volume
    grain_radius   : mean grain radius [mm]; each grain gets a radius of 0.7-1.3 times this
    composition    : mineral -> volume fraction of the grains (default Bentheimer)
    """

    def __init__(self, height=2.0, radius=1.0, voxel_size=0.02, porosity=0.23, grain_radius=0.12,
                 composition=SANDSTONE, seed=0):
        self.height, self.radius, self.voxel_size = height, radius, voxel_size
        self.porosity, self.grain_radius, self.composition = porosity, grain_radius, dict(composition)
        self.seed = seed
        self.core = self.rock = self.pore = self.connected = None
        self.co2_time = None            # time [s] at which CO2 enters each voxel (inf = never)
        self.injection_end = 0.0

    # ---- 1. geometry and rock
    def cylinder(self):
        """Boolean [z, y, x] array, True inside the cylinder."""
        nz = round(self.height / self.voxel_size)
        n = 2 * round(self.radius / self.voxel_size)
        c = (n - 1) / 2
        y, x = np.ogrid[:n, :n]
        disc = (y - c)**2 + (x - c)**2 <= (n / 2)**2
        self.core = np.broadcast_to(disc, (nz, n, n)).copy()
        return self.core

    def make_sample(self):
        """
        Fill the cylinder with overlapping spherical grains until the porosity is reached.
        Each grain is one mineral, drawn with probability equal to its volume fraction in `composition`.
        Only pores connected from the bottom face to the top face can be reached by injected CO2;
        the isolated ones stay water-filled.
        """
        core = self.core if self.core is not None else self.cylinder()
        rng = np.random.default_rng(self.seed)
        minerals = list(self.composition)
        p = np.array([self.composition[m] for m in minerals], float)
        p /= p.sum()

        self.rock = np.zeros(core.shape, np.uint8)        # mineral label of each voxel, 0 = pore or outside
        target = (1 - self.porosity) * core.sum()          # number of solid voxels wanted
        n_solid, r_mean = 0, self.grain_radius / self.voxel_size
        while n_solid < target:
            centre = rng.uniform(0, core.shape)
            r = r_mean * rng.uniform(0.7, 1.3)
            # only touch a small box around the grain (much faster than the whole array)
            box = tuple(slice(max(int(c - r), 0), min(int(c + r) + 2, n)) for c, n in zip(centre, core.shape))
            zz, yy, xx = np.ogrid[box]
            ball = (zz - centre[0])**2 + (yy - centre[1])**2 + (xx - centre[2])**2 <= r**2
            new = ball & core[box] & (self.rock[box] == 0)
            self.rock[box][new] = MINERAL_LABEL[rng.choice(minerals, p=p)]
            n_solid += new.sum()

        self.pore = core & (self.rock == 0)
        clusters, _ = ndi.label(self.pore)
        through = np.intersect1d(clusters[0], clusters[-1])
        through = through[through > 0]                     # pore clusters touching both faces
        if through.size == 0:
            raise RuntimeError("the pore space does not connect the bottom to the top; raise the porosity or change the seed")
        self.connected = np.isin(clusters, through)
        self._seq = None
        return self.rock

    @property
    def measured_porosity(self):
        return self.pore.sum() / self.core.sum()

    # ---- 2. fluids
    def _invasion_order(self):
        """
        Order in which a non-wetting phase (CO2) fills the water-wet pores when it is pushed in from the
        bottom: invasion percolation with PoreSpy (QBIP), the widest reachable opening first, stopped
        at breakthrough at the top. Quasi-static: no pressure or velocity field.
        """
        if self._seq is None:
            import porespy as ps
            inlet = np.zeros_like(self.connected)
            inlet[0] = self.connected[0]
            outlet = np.zeros_like(self.connected)
            outlet[-1] = self.connected[-1]
            seq = np.asarray(ps.simulations.injection(im=self.connected, inlets=inlet, outlets=outlet,
                                                      method="qbip").im_seq)
            seq = np.where(self.connected & (seq > 0), seq, 0).astype(np.int64)
            at_outlet = seq[-1][seq[-1] > 0]
            if at_outlet.size:
                seq[seq > at_outlet.min()] = 0             # stop at breakthrough
            self._seq = seq
            counts = np.bincount(seq.ravel())
            counts[0] = 0                                  # step 0 = never filled
            self._cum = np.cumsum(counts)                  # _cum[s] = voxels filled after step s
        return self._seq, self._cum

    def set_start_water_saturation(self, S_w0=1.0):
        """
        Water saturation at t = 0. The pores start 100 % water; for S_w0 < 1 the core is drained to S_w0 by
        gas pushed in from the bottom (the usual way to set an initial saturation in the lab). The gas is
        taken to be CO2, already in place when the injection starts.
        """
        seq, cum = self._invasion_order()
        n_gas = (1 - S_w0) * self.pore.sum()
        if n_gas > cum[-1]:
            raise ValueError(f"S_w0 can be at lowest {1 - cum[-1] / self.pore.sum():.2f} "
                             "(below that the gas has broken through at the top)")
        self.start_step = int(np.searchsorted(cum, n_gas))
        self.co2_time = np.full(self.core.shape, np.inf)
        self.co2_time[(seq > 0) & (seq <= self.start_step)] = 0.0
        self.injection_end = 0.0

    def inject_co2(self, start=0.0, duration=300.0, pump_rate=0.1):
        """
        Pump CO2 in from the bottom at a constant rate [uL/min] from `start` for `duration` seconds.
        CO2 keeps displacing water in the invasion order. Each invasion step is complete once its volume has
        been pumped in, which turns the order into a time for every voxel. After breakthrough at the top
        the CO2 flows straight through and the saturation no longer changes.
        """
        if self.co2_time is None:
            self.set_start_water_saturation(1.0)
        seq, cum = self._invasion_order()
        voxel_volume = self.voxel_size**3                                  # [mm^3] = [uL]
        t_step = start + (cum - cum[self.start_step]) * voxel_volume / pump_rate * 60
        t_vox = t_step[seq]
        injected = (seq > self.start_step) & (t_vox <= start + duration)
        self.co2_time[injected] = t_vox[injected]
        self.injection_end = start + duration
        self.breakthrough_time = t_step[-1]

    def labels_at(self, t):
        """Material label of every voxel at time t [s]: minerals, water and CO2."""
        labels = np.where(self.pore, WATER, self.rock).astype(np.uint8)
        if self.co2_time is not None:
            labels[self.co2_time <= t] = CO2
        return labels

    def mu_map(self, t, energy):
        """Linear attenuation [1/mm] of every voxel at time t [s] and energy [keV]."""
        return mu_map(self.labels_at(t), energy)


# ---------------------------------------------------------------------------------------------------------
# Hydrate growth
# ---------------------------------------------------------------------------------------------------------
class HydrateGrowth:
    """
    CO2 hydrate formation in a CoreSample, after the CO2 injection.

    From `start` the core is held at `pressure` [MPa] and `temperature` [K]. If that is inside the hydrate
    stability zone, the first crystal appears after a random induction time. Hydrate then nucleates at a few
    points on the water-CO2 interface, spreads quickly along the interface as a film and thickens slowly into
    the water (diffusion limited, thickness ~ sqrt(t)). Both rates grow with the subcooling. Hydrate takes
    ~29 % more volume than the water it consumes, so the extra volume grows into the CO2 next to it. The pump
    keeps the pressure, so the CO2 used up is replaced.

    pressure, temperature : conditions during growth [MPa], [K]
    n_nuclei              : nucleation sites on the interface
    film_rate             : speed of the film along the interface per kelvin of subcooling [mm/s/K] (illustrative)
    thickening_rate       : thickening constant per kelvin, thickness^2 = rate * dT * t [mm^2/s/K] (illustrative)
    max_conversion        : largest fraction of the water that can turn into hydrate
    induction_time        : waiting time [s] before nucleation; None = draw it from the nucleation model
    J0, B                 : nucleation rate model J = J0 exp(-B / (T dT^2)) [1/h], [K^3] (illustrative)
    """

    # quadruple points (T [K], P [MPa]); the equilibrium curve between them is a straight line in ln P vs 1/T
    Q1 = (271.8, 1.04)          # hydrate, ice, liquid water, CO2 gas
    Q2 = (283.0, 4.50)          # hydrate, liquid water, CO2 gas, liquid CO2

    def __init__(self, core, pressure=3.0, temperature=274.15, n_nuclei=50, film_rate=1e-3, thickening_rate=1e-6,
                 max_conversion=0.9, induction_time=None, J0=1.0, B=1.2e4, seed=0):
        self.core = core
        self.pressure, self.temperature = pressure, temperature
        self.n_nuclei, self.film_rate, self.thickening_rate = n_nuclei, film_rate, thickening_rate
        self.max_conversion, self.J0, self.B = max_conversion, J0, B
        self.rng = np.random.default_rng(seed)
        self._induction = induction_time
        self.hydrate_time = np.full(core.core.shape, np.inf)   # time [s] at which each voxel becomes hydrate

    # ---- thermodynamics
    @classmethod
    def _fit(cls):
        b = (np.log(cls.Q2[1]) - np.log(cls.Q1[1])) / (1 / cls.Q2[0] - 1 / cls.Q1[0])
        return np.log(cls.Q1[1]) - b / cls.Q1[0], b

    @classmethod
    def equilibrium_pressure(cls, T):
        """Hydrate equilibrium pressure [MPa] at temperature T [K]."""
        a, b = cls._fit()
        return np.exp(a + b / T)

    @classmethod
    def equilibrium_temperature(cls, P):
        """Hydrate equilibrium temperature [K] at pressure P [MPa]."""
        a, b = cls._fit()
        return b / (np.log(P) - a)

    def subcooling(self):
        """Driving force: equilibrium temperature at the pressure minus the actual temperature [K]."""
        return self.equilibrium_temperature(self.pressure) - self.temperature

    def is_stable(self):
        return self.subcooling() > 0

    def induction_time(self):
        """Time [s] from entering the stability zone until the first crystal. Random (exponential), as nucleation is."""
        if self._induction is not None:
            return self._induction
        if not self.is_stable():
            return np.inf
        rate = self.J0 * np.exp(-self.B / (self.temperature * self.subcooling()**2))   # [1/h]
        self._induction = self.rng.exponential(1 / rate) * 3600
        return self._induction

    # ---- geometry of growth
    def interface(self, labels):
        """Water voxels that touch CO2: where hydrate can nucleate."""
        return (labels == WATER) & ndi.binary_dilation(labels == CO2)

    def nucleate(self, labels):
        """Pick the nucleation sites at random on the water-CO2 interface."""
        sites = np.flatnonzero(self.interface(labels))
        nuclei = np.zeros(labels.shape, bool)
        nuclei.flat[self.rng.choice(sites, min(self.n_nuclei, sites.size), replace=False)] = True
        return nuclei

    @staticmethod
    def pore_distance(source, medium):
        """
        Distance [voxels] from `source` to every voxel of `medium`, moving only through `medium`
        (so the growth front goes around grains, not through them). inf where it cannot get to.
        """
        step = ndi.generate_binary_structure(3, 2)               # 18 neighbours
        dist = np.full(source.shape, np.inf)
        dist[source] = 0
        front, reached, k = source.copy(), source.copy(), 0
        while front.any():
            k += 1
            front = ndi.binary_dilation(front, step) & medium & ~reached
            dist[front] = k
            reached |= front
        return dist

    def arrival_time(self, labels, nuclei):
        """
        Time [s] after nucleation at which the growth front reaches each voxel, as the sum of
          film spreading along the interface at constant speed:  d_nuc / (film_rate * dT)
          thickening, limited by CO2 diffusing through the hydrate already formed, so the
          thickness grows as sqrt(t):                             d_int^2 / (thickening_rate * dT)
        d_nuc = distance from the nearest nucleus, d_int = distance from the water-CO2 interface [mm],
        both measured through the pore space.
        """
        water, co2 = labels == WATER, labels == CO2
        d_int = np.where(water, self.pore_distance(co2, water), self.pore_distance(water, co2)) * self.core.voxel_size
        d_nuc = self.pore_distance(nuclei, water | co2) * self.core.voxel_size
        dT = self.subcooling()
        return d_nuc / (self.film_rate * dT) + d_int**2 / (self.thickening_rate * dT)

    # ---- the growth itself
    def grow(self, start=None, duration=3600.0):
        """
        Grow hydrate from `start` (default: end of the CO2 injection) for `duration` seconds.
        Water voxels turn into hydrate in the order the front reaches them. Because of the volume expansion,
        every water voxel converted makes room for (EXPANSION - 1) CO2 voxels next to the front to turn into
        hydrate too. Growth stops at max_conversion, or earlier if there is no CO2 left to expand into.
        """
        start = self.core.injection_end if start is None else start
        if start < self.core.injection_end:
            raise ValueError("start hydrate growth after the CO2 injection has ended")
        self.start, self.end = start, start + duration
        self.hydrate_time[:] = np.inf
        if not self.is_stable():
            print(f"{self.pressure} MPa, {self.temperature} K is outside the hydrate stability zone: no hydrate")
            return
        labels = self.core.labels_at(start)
        if not self.interface(labels).any():
            print("no water-CO2 interface: no hydrate")
            return

        self.nucleation_time = start + self.induction_time()
        t = self.nucleation_time + self.arrival_time(labels, self.nucleate(labels))

        water = np.flatnonzero(labels == WATER)
        co2 = np.flatnonzero(labels == CO2)
        f = EXPANSION - 1
        n_w = min(int(self.max_conversion * water.size), int(co2.size / f))
        water = water[np.argsort(t.flat[water], kind="stable")][:n_w]
        t_w = t.flat[water]
        co2 = co2[np.argsort(t.flat[co2], kind="stable")][:int(f * n_w)]
        # CO2 voxel k needs ceil((k + 1) / f) water voxels converted first
        need = np.ceil((np.arange(co2.size) + 1) / f).astype(int) - 1
        t_c = np.maximum(t.flat[co2], t_w[np.minimum(need, n_w - 1)])

        self.hydrate_time.flat[water] = t_w
        self.hydrate_time.flat[co2] = t_c
        self.hydrate_time[self.hydrate_time > self.end] = np.inf

    def labels_at(self, t):
        """Material label of every voxel at time t [s], including hydrate."""
        labels = self.core.labels_at(t)
        labels[self.hydrate_time <= t] = HYDRATE
        return labels

    def mu_map(self, t, energy):
        """
        Linear attenuation [1/mm] of every voxel at time t [s] and energy [keV]:
        the core's map, with the voxels that have formed hydrate changed to the attenuation of hydrate.
        """
        mu_vox = self.core.mu_map(t, energy)
        mu_vox[self.hydrate_time <= t] = mu("hydrate", energy)
        return mu_vox


# ---------------------------------------------------------------------------------------------------------
# 3D plot
# ---------------------------------------------------------------------------------------------------------
def plot_3d(ax, labels, voxel_size, phases=((CO2, "dimgrey", 0.6), (HYDRATE, "tab:red", 0.35)), step=2):
    """
    Draw the surfaces of the chosen phases on a 3D matplotlib axis, inside an outline of the core.
    Grains and water are left out, since they fill most of the core and would hide the rest.
    phases: (label, colour, opacity) for each phase to draw.
    step  : surface resolution in voxels; 1 = full detail but slow, 2 is about 4 times faster.
    """
    from skimage.measure import marching_cubes
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    nz, ny, nx = labels.shape
    for lab, colour, alpha in phases:
        mask = np.pad(labels == lab, 1)                       # pad so the surfaces close at the core edges
        if not mask.any():
            continue
        verts, faces, _, _ = marching_cubes(mask.astype(np.float32), 0.5, step_size=step)
        verts = (verts - 1)[:, ::-1] * voxel_size             # [z, y, x] -> (x, y, z) in mm
        ax.add_collection3d(Poly3DCollection(verts[faces], facecolors=colour, alpha=alpha, linewidth=0, shade=True))

    # outline of the cylinder: top and bottom circles and four lines along the sides
    r, c = nx / 2 * voxel_size, nx / 2 * voxel_size
    phi = np.linspace(0, 2 * np.pi, 100)
    for z in (0, nz * voxel_size):
        ax.plot(c + r * np.cos(phi), c + r * np.sin(phi), z, color="k", lw=0.8)
    for p in np.linspace(0, 2 * np.pi, 4, endpoint=False):
        ax.plot([c + r * np.cos(p)] * 2, [c + r * np.sin(p)] * 2, [0, nz * voxel_size], color="k", lw=0.5)
    ax.set(xlim=(0, nx * voxel_size), ylim=(0, ny * voxel_size), zlim=(0, nz * voxel_size),
           xlabel="x [mm]", ylabel="y [mm]", zlabel="z [mm]")
    ax.set_box_aspect((nx, ny, nz))


# ---------------------------------------------------------------------------------------------------------
# Example: the core in 3D at three times, and the saturations over time
# ---------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    from datetime import datetime
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")                  # only save to file, no window
    import matplotlib.pyplot as plt

    t0 = time.time()
    core = CoreSample(height=2.0, radius=1.0, voxel_size=0.02, porosity=0.23, grain_radius=0.12)
    core.cylinder()
    core.make_sample()
    print(f"rock {core.rock.shape}, porosity {core.measured_porosity:.3f}  ({time.time() - t0:.1f} s)")

    core.set_start_water_saturation(1.0)
    core.inject_co2(start=0, duration=300, pump_rate=0.1)
    print(f"CO2 injected, S = {saturations(core.labels_at(300))}  ({time.time() - t0:.1f} s)")

    hyd = HydrateGrowth(core, pressure=3.0, temperature=274.15, induction_time=120)
    hyd.grow(start=300, duration=3600)
    print(f"subcooling {hyd.subcooling():.1f} K, nucleation at {hyd.nucleation_time:.0f} s  ({time.time() - t0:.1f} s)")

    # every run gets its own folder, runs/<date>_<time>, next to this file
    run_dir = Path(__file__).parent / "runs" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    times = [150, 300, 3900]           # times shown in the 3D figure
    time_names = ["during injection", "end of injection", "end of growth"]

    # 1. the core in 3D at the three times
    fig = plt.figure(figsize=(16, 6))
    for i, t in enumerate(times):
        ax = fig.add_subplot(1, len(times), i + 1, projection="3d")
        plot_3d(ax, hyd.labels_at(t), core.voxel_size)
        ax.set_title(f"t = {t} s")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("dimgrey", "tab:red")]
    fig.legend(handles, ["CO$_2$", "hydrate"], loc="lower left")
    fig.subplots_adjust(left=0.02, right=0.96, bottom=0.08, top=0.92, wspace=0.15)
    fig.savefig(run_dir / "hydrate_3d.png", dpi=150)
    plt.close(fig)

    # 2. pore saturations over time
    fig, ax = plt.subplots(figsize=(7, 4.5))
    t_curve = np.linspace(0, times[-1], 80)
    S = [saturations(hyd.labels_at(t)) for t in t_curve]
    for name, colour in (("water", "tab:blue"), ("CO2", "grey"), ("hydrate", "tab:red")):
        ax.plot(t_curve, [s[name] for s in S], color=colour, label=name)
    for t, name in zip(times, time_names):
        ax.axvline(t, color="k", ls=":", lw=1)
        ax.text(t, 0.5, f"{name}, t = {t} s", rotation=90, ha="right", va="center", fontsize=8,
                transform=ax.get_xaxis_transform(), bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set(xlabel="time [s]", ylabel="saturation", title="Pore saturations (dotted: times in the 3D figure)")
    ax.legend(loc="upper center", ncol=3)
    fig.tight_layout()
    fig.savefig(run_dir / "saturations.png", dpi=150)
    plt.close(fig)
    print(f"saved figures in {run_dir}")
