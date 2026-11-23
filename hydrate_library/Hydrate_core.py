# Hydrate phantom: CO2 injected into brine-saturated Bentheimer sandstone, hydrate forms, CT attenuation per time step.
# ---------------------------------------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------------------------------------
# Every voxel stores a VOLUME FRACTION for each phase (array shape: n_phases x nz x ny x nx, float32).
# Micro-CT scale: fractions are mostly 0 or 1. Medical CT scale: mixed voxels (partial volume) come for free.
# mu_map is then a weighted sum over phases, so no labels need translating later.
# Memory check: 8 phases x 200^3 voxels x 4 bytes = 256 MB. Start small (e.g. 1 x 1 mm core at 10 um voxels).
#
# Tested with porespy 3.3 and xraylib 4.x.

import numpy as np
from scipy import ndimage
import porespy as ps
import xraylib
from CoolProp.CoolProp import PropsSI

# ---------------------------------------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------------------------------------
# name -> ({chemical formula: weight fraction}, density [g/cm^3])
MATERIALS = {
    "quartz":    ({"SiO2": 1.0},           2.65),
    "feldspar":  ({"KAlSi3O8": 1.0},       2.56),
    "kaolinite": ({"Al2Si2O5(OH)4": 1.0},  2.60),
    "brine":     ({"H2O": 0.965, "NaCl": 0.035}, 1.025),  # TODO: set to the salinity used in the experiments
    "fluid":     ({"CO2": 1.0},            0.07),   # injected fluid; composition and density set by setup_materials
    "hydrate":   ({"CO2(H2O)6.08": 1.0},   1.09),
    "ice":       ({"H2O": 1.0},            0.917),
    "outside":   ({"N": 0.78, "O": 0.22},  0.0012), # air around the core; swap for the holder material if needed
}
PHASES = []                                    # order along axis 0 of the state array, set by setup_materials
IDX = {}                                       # phase name -> that position, e.g. state[IDX["hydrate"]]
MINERALS = []                                  # set by setup_materials

# Default rock: Bentheimer grain composition, XRD weight fractions (Peksa et al. 2015, Table 1).
# Clay = kaolinite 2.50 + montmorillonite 0.18; the remaining 0.7 wt% (carbonates, oxides, pyrite) is left out.
ROCK_WT = {"quartz": 0.9170, "feldspar": 0.0486, "kaolinite": 0.0268}
ROCK = {}                                      # volume fractions, filled by setup_materials
FLUID = {"name": "CO2"}                        # CoolProp name of the injected fluid


def setup_materials(cfg):
    "Read rock and injected fluid from the config, and rebuild PHASES and IDX. Call before making the sample."
    # extra or overridden materials: name -> {formula: {...}, density: ...}
    for name, m in (cfg.get("materials") or {}).items():
        MATERIALS[name] = (m["formula"], m["density"])

    # rock: weight fractions of minerals, normalised to 1
    rock_wt = cfg.get("rock", {}).get("minerals", ROCK_WT)
    for m in rock_wt:
        if m not in MATERIALS:
            raise KeyError(f"mineral '{m}' is not in MATERIALS; add it under 'materials' in the config")
    total = sum(rock_wt.values())
    vol = {m: (w / total) / MATERIALS[m][1] for m, w in rock_wt.items()}
    ROCK.clear()
    ROCK.update({m: v / sum(vol.values()) for m, v in vol.items()})
    MINERALS[:] = list(rock_wt)

    # injected fluid: CoolProp name and chemical formula; density is set later from P and T
    fluid = cfg.get("fluid", {})
    FLUID["name"] = fluid.get("name", "CO2")
    MATERIALS["fluid"] = (fluid.get("formula", {"CO2": 1.0}), MATERIALS["fluid"][1])

    # phases: the chosen minerals, then the pore phases (in place, so other modules see the change)
    PHASES[:] = MINERALS + ["brine", "fluid", "hydrate", "ice", "outside"]
    IDX.clear()
    IDX.update({name: i for i, name in enumerate(PHASES)})


setup_materials({})                            # defaults, so the module also works without a config


# ---------------------------------------------------------------------------------------------------------
# Core sample: geometry, rock, water, injection
# ---------------------------------------------------------------------------------------------------------
class CoreSample:
    def __init__(self, height_mm, radius_mm, voxel_size_um, porosity, S_w=1.0,
                 flow_rate=1e-3, inlet_radius_mm=None, residual_brine=0.1, blobiness=1, seed=0):
        self.height_mm = height_mm
        self.radius_mm = radius_mm
        self.voxel_size_um = voxel_size_um     # physical scale, needed because mu is per length
        self.porosity = porosity
        self.S_w = S_w
        self.flow_rate = flow_rate             # mm^3 CO2 per second, at experiment P and T
        self.inlet_radius_mm = inlet_radius_mm if inlet_radius_mm is not None else radius_mm / 4
        self.residual_brine = residual_brine   # brine film left in a voxel after CO2 invades it
        self.blobiness = blobiness             # larger value = smaller grains in ps.generators.blobs
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.voxel_volume_mm3 = (voxel_size_um * 1e-3) ** 3
        self.shape = None                      # (nz, ny, nx), set by make_cylinder
        self.inside = None                     # bool mask of the cylinder
        self.solid = None                      # bool mask of grains
        self.pore = None                       # bool mask of pore space inside the cylinder
        self.pore_size = None                  # local pore radius per voxel [um], 0 outside the pores
        self.invasion_order = None             # flat indices of pore voxels, in the order CO2 reaches them
        self.n_invaded = 0                     # how many voxels in invasion_order CO2 has reached so far
        self.injected_mm3 = 0.0
        self.state = None                      # volume fractions, shape (n_phases, nz, ny, nx)

    def make_cylinder(self):
        "Voxel grid from height, radius and voxel size; mask of voxels inside the cylinder"
        vs_mm = self.voxel_size_um * 1e-3
        nz = int(round(self.height_mm / vs_mm))
        n = int(round(2 * self.radius_mm / vs_mm))
        self.shape = (nz, n, n)

        c = (n - 1) / 2
        y, x = np.ogrid[:n, :n]
        disc = (x - c) ** 2 + (y - c) ** 2 <= (n / 2) ** 2
        self.inside = np.broadcast_to(disc, self.shape).copy()

        self.state = np.zeros((len(PHASES), *self.shape), dtype=np.float32)
        self.state[IDX["outside"]][~self.inside] = 1.0
        return self.state

    def make_rock(self, per_grain=False):
        "Grain structure with PoreSpy at the target porosity, then assign minerals to grains"
        pore_im = ps.generators.blobs(shape=list(self.shape), porosity=self.porosity,
                                      blobiness=self.blobiness, seed=self.seed, periodic=False)
        self.solid = ~pore_im & self.inside
        self.pore = pore_im & self.inside

        if not per_grain:
            # every solid voxel gets the Bentheimer mixture (good enough for medical CT and a first version)
            for m in MINERALS:
                self.state[IDX[m]][self.solid] = ROCK[m]
        else:
            # split the solid into grains (watershed), give each grain one mineral
            grains = ps.filters.snow_partitioning(im=self.solid).regions * self.solid
            n_grains = grains.max()
            p = np.array([ROCK[m] for m in MINERALS])
            mineral_of_grain = self.rng.choice(len(MINERALS), size=n_grains + 1, p=p)
            for k, m in enumerate(MINERALS):
                self.state[IDX[m]][self.solid & (mineral_of_grain[grains] == k)] = 1.0
        return self.state

    def compute_pore_size(self):
        "Local pore size for every pore voxel"
        # local_thickness gives the radius (in voxels) of the largest sphere that fits and covers each voxel
        self.pore_size = ps.filters.local_thickness(self.pore) * self.voxel_size_um
        return self.pore_size

    def set_water_saturation(self):
        "Fill the pore space with brine, and with CO2 for the part 1 - S_w"
        self.state[IDX["brine"]][self.pore] = 1.0
        if self.S_w < 1.0:
            if self.invasion_order is None:
                self.compute_invasion_order()
            n_gas = int((1 - self.S_w) * self.pore.sum())
            self._invade(n_gas)
        return self.state

    def compute_invasion_order(self):
        "Invasion percolation from an inlet disc at the bottom slice, run once before injecting"
        inlets = np.zeros(self.shape, dtype=bool)
        n = self.shape[1]
        c = (n - 1) / 2
        y, x = np.ogrid[:n, :n]
        r_vox = self.inlet_radius_mm / (self.voxel_size_um * 1e-3)
        inlets[0] = ((x - c) ** 2 + (y - c) ** 2 <= r_vox ** 2) & self.pore[0]

        # capillary entry pressure follows from pore size in the image, so Leverett J is not needed
        res = ps.simulations.injection(im=self.pore, inlets=inlets)
        seq = res.im_seq.ravel()
        reached = np.flatnonzero(seq > 0)                    # voxels CO2 can reach at all
        self.invasion_order = reached[np.argsort(seq[reached], kind="stable")]
        return self.invasion_order

    def _invade(self, n_new):
        "Turn the next n_new voxels in invasion order from brine into CO2, leaving a residual brine film"
        start, stop = self.n_invaded, min(self.n_invaded + n_new, len(self.invasion_order))
        idx = self.invasion_order[start:stop]
        brine = self.state[IDX["brine"]].reshape(-1)         # views, so writing changes self.state
        fluid = self.state[IDX["fluid"]].reshape(-1)
        moved = np.clip(brine[idx] - self.residual_brine, 0, None)
        brine[idx] -= moved
        fluid[idx] += moved
        self.n_invaded = stop

    def inject(self, t, dt=1.0, consumed_mm3=0.0):
        "Place this step's injected CO2 volume, then let the connected gas refill free space in invaded voxels"
        self.injected_mm3 += self.flow_rate * dt
        available = self.injected_mm3 - consumed_mm3
        target = int(available / (self.voxel_volume_mm3 * (1 - self.residual_brine)))  # voxels' worth of CO2 injected so far
        self._invade(max(target - self.n_invaded, 0))

        # gas is connected to the inlet at constant pressure, so CO2 used by hydrate gets replaced
        idx = self.invasion_order[:self.n_invaded]
        flat = self.state.reshape(len(PHASES), -1)
        free = 1.0 - flat[:, idx].sum(axis=0)
        flat[IDX["fluid"], idx] += np.clip(free, 0, None)
        return self.state


# ---------------------------------------------------------------------------------------------------------
# Hydrate formation
# ---------------------------------------------------------------------------------------------------------
# Bulk CO2 hydrate equilibrium (hydrate + liquid water + CO2 gas), valid about 273-283 K.
# APPROXIMATE values for a first version. TODO: replace with tabulated data from a source you cite (e.g. Sloan & Koh 2008).
_EQ_T = np.array([273.7, 275.1, 276.5, 277.6, 279.2, 280.3, 281.5, 282.6])   # K
_EQ_P = np.array([1.39, 1.63, 1.95, 2.23, 2.71, 3.09, 3.62, 4.24])            # MPa
_B, _A = np.polyfit(1 / _EQ_T, np.log(_EQ_P), 1)                             # ln P = A + B / T


class HydrateFormation:
    HYDRATION_NUMBER = 6.08
    M_H2O, M_CO2 = 18.015, 44.01               # g/mol
    # approximate constants, TODO: check against literature
    SIGMA_HW = 0.027                           # hydrate-water interfacial tension [J/m^2]
    DH_DISS = 65e3                             # dissociation enthalpy [J per mol hydrate]
    DT_PER_WT_NACL = 0.4                       # equilibrium T shift per wt% NaCl [K], crude linear placeholder
    GAMMA_CO2_BRINE = 0.03                     # CO2-brine interfacial tension [N/m]
    CONTACT_ANGLE_DEG = 30                     # water-wet sandstone
    PERMEABILITY_MD = 2500                     # Bentheimer, order of 1-3 D; TODO: use the measured value of your core
    J_ENTRY = 0.2                              # Leverett J at the entry pressure (placeholder)
    BC_LAMBDA = 2.0                            # Brooks-Corey pore size index (placeholder, larger = more uniform pores)
    S_WR = 0.1                                 # irreducible brine saturation (placeholder)

    def __init__(self, sample, formation_start, timestep, temperature, pressure, salinity,
                 morphology_type="grain_coating", memory_effect=False, memory_factor=0.3,
                 growth_rate=0.1, nucleation_probability=0.01, seed=0):
        self.sample = sample                   # S_w and pore sizes are read from here, not passed again
        self.formation_start = formation_start # induction time [s]
        self.timestep = timestep               # [s]
        self.temperature = temperature         # K
        self.pressure = pressure               # MPa
        self.salinity = salinity               # wt% NaCl
        self.morphology_type = morphology_type # "grain_coating" or "pore_filling"
        self.memory_effect = memory_effect
        self.memory_factor = memory_factor     # induction time multiplier when the water has hydrate memory
        self.growth_rate = growth_rate         # fraction of available brine converted per step
        self.nucleation_probability = nucleation_probability  # per candidate voxel per step
        self.rng = np.random.default_rng(seed)
        self.nucleated = np.zeros(sample.shape, dtype=bool)
        self.co2_consumed_mm3 = 0.0            # CO2 gas volume built into hydrate, for mass bookkeeping
        self.brine_displaced_mm3 = 0.0         # brine pushed out by hydrate expansion
        self.stable = None

    def set_temperature_and_pressure(self, temperature=None, pressure=None):
        "Set the temperature and pressure of the sample, and tell where hydrate formation can occur"
        if temperature is not None:
            self.temperature = temperature
        if pressure is not None:
            self.pressure = pressure

        # CO2 density at the current P and T, used by hydrate_conversion and material_mu
        rho = PropsSI("D", "T", self.temperature, "P", self.pressure * 1e6, FLUID["name"]) / 1000
        MATERIALS["fluid"] = (MATERIALS["fluid"][0], rho)
        T_eq_bulk = _B / (np.log(self.pressure) - _A)        # equilibrium T at this pressure
        T_eq = T_eq_bulk - self.water_salinity() - self.pore_size_distribution(T_eq_bulk)
        self.stable = self.sample.pore & (self.temperature < T_eq)
        return self.stable

    # effects of the porous medium
    def pore_size_distribution(self, T_eq_bulk=None, bins=30):
        "Gibbs-Thomson shift of the equilibrium T per voxel; with no T given, the pore size histogram"
        r_um = self.sample.pore_size
        if T_eq_bulk is None:
            return np.histogram(r_um[self.sample.pore], bins=bins)
        # dT = 2 sigma T / (r rho_h dH), dH per kg hydrate
        M_h = (self.M_CO2 + self.HYDRATION_NUMBER * self.M_H2O) * 1e-3        # kg/mol
        dH_kg = self.DH_DISS / M_h
        rho_h = MATERIALS["hydrate"][1] * 1e3
        r_m = np.where(r_um > 0, r_um * 1e-6, np.inf)
        return 2 * self.SIGMA_HW * T_eq_bulk / (r_m * rho_h * dH_kg)
        # about 0.03 K at r = 1 um, so negligible in Bentheimer; only matters for pores below ~0.1 um

    def nucleation_sites(self, t):
        "Determine the nucleation sites for hydrate formation"
        
        if t * self.timestep < self.memory_effects():
            return np.zeros(self.sample.shape, dtype=bool)
        s = self.sample.state
        brine = s[IDX["brine"]] > 0
        near_gas = ndimage.binary_dilation(s[IDX["fluid"]] > 0)
        candidates = self.stable & brine & near_gas & ~self.nucleated
        new = candidates & (self.rng.random(self.sample.shape) < self.nucleation_probability)
        self.nucleated |= new
        return new

    def morphological_effects(self):
        "Where hydrate grows next, as a weight 0-1 per voxel, according to the morphology type"
        s = self.sample.state
        brine = s[IDX["brine"]] > 0
        near_gas = ndimage.binary_dilation(s[IDX["fluid"]] > 0)
        has_hydrate = self.nucleated | (s[IDX["hydrate"]] > 0)
        front = self.stable & brine & near_gas & ndimage.binary_dilation(has_hydrate)

        if self.morphology_type == "grain_coating":
            # only next to grains, so hydrate grows as a shell on the grain surface
            return (front & ndimage.binary_dilation(self.sample.solid)).astype(np.float32)
        if self.morphology_type == "pore_filling":
            # pore centres first: weight by pore size
            r = self.sample.pore_size
            return np.where(front, r / r.max(), 0).astype(np.float32)
        # contact cementing, patchy, load bearing, supporting matrix: later
        raise ValueError(f"unknown morphology_type {self.morphology_type}")

    # effects of the pore water
    def hydrate_saturation(self):
        "Hydrate saturation of the sample: hydrate volume / pore volume"
        s = self.sample.state
        return float(s[IDX["hydrate"]].sum() / self.sample.pore.sum())

    def memory_effects(self):
        "Memory effects of the sample on hydrate formation: a shorter induction time, nothing more for now"
        return self.formation_start * (self.memory_factor if self.memory_effect else 1.0)

    def water_salinity(self):
        "Effects of the water salinity: shift of the equilibrium temperature [K]"
        # TODO: replace with a proper correlation (e.g. Hu-Lee-Sum 2017)
        # later: salt stays in the remaining brine, so salinity rises as hydrate forms
        return self.DT_PER_WT_NACL * self.salinity

    # capillary effects
    def leverett_J(self, S_w):
        "Dimensionless J(S_w), Brooks-Corey form: J = J_e * S_e^(-1/lambda)"
        S_e = np.clip((S_w - self.S_WR) / (1 - self.S_WR), 1e-3, 1.0)   # effective saturation, clipped to avoid infinity
        return self.J_ENTRY * S_e ** (-1 / self.BC_LAMBDA)

    def capillary_pressure(self, per_slice=True):
        "Capillary pressure from the Leverett J-function [kPa]: Pc = sigma cos(theta) sqrt(phi / k) J(S_w)"
        s = self.sample.state
        pore = self.sample.pore
        axes = (1, 2) if per_slice else None
        # S_w per slice along the core (a saturation profile, like the medical CT) or for the whole sample
        S_w = s[IDX["brine"]].sum(axis=axes) / np.maximum(pore.sum(axis=axes), 1)
        phi = pore.sum() / self.sample.inside.sum()
        k_m2 = self.PERMEABILITY_MD * 9.869e-16
        Pc = self.GAMMA_CO2_BRINE * np.cos(np.radians(self.CONTACT_ANGLE_DEG)) * np.sqrt(phi / k_m2) \
            * self.leverett_J(S_w)
        return Pc * 1e-3

    def hydrate_conversion(self, V_water):
        "Mass-conserving conversion: a volume of water -> volume of hydrate formed and volume of CO2 consumed"
        n_w = self.HYDRATION_NUMBER
        m_w = V_water * 1.0                                # g (pure water part; TODO: salt stays in the brine)
        n = m_w / (n_w * self.M_H2O)                       # mol hydrate
        V_h = n * (self.M_CO2 + n_w * self.M_H2O) / MATERIALS["hydrate"][1]
        V_co2 = n * self.M_CO2 / MATERIALS["fluid"][1]
        return V_h, V_co2                                  # V_h is about 1.29 x V_water

    # Growth
    def growth(self, t):
        "The hydrate growth itself, one time step"
        if self.stable is None:
            self.set_temperature_and_pressure()
        self.nucleation_sites(t)
        w = self.morphological_effects()

        s = self.sample.state
        brine, co2, hyd = s[IDX["brine"]], s[IDX["fluid"]], s[IDX["hydrate"]]
        r_h, r_c = self.hydrate_conversion(1.0)            # hydrate and CO2 volume per unit water

        # cap so the voxel never overfills: brine left after conversion and displacement stays >= 0
        dV_w = np.minimum(self.growth_rate * w * brine, brine / r_h)

        # the CO2 that goes into the hydrate comes from the connected gas, which inject() keeps replacing
        self.co2_consumed_mm3 += float((r_c * dV_w).sum()) * self.sample.voxel_volume_mm3

        brine -= dV_w
        hyd += r_h * dV_w

        # hydrate takes more room than the water it came from; it pushes out gas first, then brine
        excess = np.clip(s.sum(axis=0) - 1.0, 0, None)
        pushed_gas = np.minimum(excess, co2)
        co2 -= pushed_gas
        pushed_brine = np.minimum(excess - pushed_gas, brine)
        brine -= pushed_brine
        self.brine_displaced_mm3 += float(pushed_brine.sum()) * self.sample.voxel_volume_mm3
        return s

    # later, once the above runs end to end
    def find_impurities(self):
        return None

    # near wellbore effects
    def new_porosity(self):
        "Porosity left open to flow: pore voxels minus the hydrate fraction"
        return None

    def relative_permeability(self):
        return None


# ---------------------------------------------------------------------------------------------------------
# Attenuation
# ---------------------------------------------------------------------------------------------------------
def material_mu(E_keV):
    "Linear attenuation coefficient [1/cm] of each phase at energy E"
    mu = np.zeros(len(PHASES), dtype=np.float32)
    for i, name in enumerate(PHASES):
        composition, density = MATERIALS[name]
        mass_att = sum(w * xraylib.CS_Total_CP(formula, E_keV) for formula, w in composition.items())  # cm^2/g
        mu[i] = density * mass_att
    return mu    # in PHASES order


def mu_map(state, E_keV):
    "Attenuation per voxel: weighted sum of phase fractions [1/cm]"
    # polychromatic later: call for each energy bin and weight by the tube spectrum
    return np.tensordot(material_mu(E_keV), state, axes=1)


# ---------------------------------------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------------------------------------
def run(sample, hydrate, n_steps, E_keV, save_every=1):
    "Inject and grow hydrate; return the phase fractions and the mu map at each saved step (the 4D phantom)"
    states, mus = [], []
    dt = hydrate.timestep
    if sample.invasion_order is None:
        sample.compute_invasion_order()
    hydrate.set_temperature_and_pressure()
    for t in range(n_steps):
        sample.inject(t, dt, consumed_mm3=hydrate.co2_consumed_mm3)
        hydrate.growth(t)
        if t % save_every == 0:
            states.append(sample.state.copy())
            mus.append(mu_map(sample.state, E_keV))
    
    return states, mus


if __name__ == "__main__":
    # small test case: 1 mm tall, 0.5 mm radius core at 10 um voxels
    core = CoreSample(height_mm=1.0, radius_mm=0.5, voxel_size_um=10, porosity=0.23,
                      flow_rate=2e-3, blobiness=1)
    core.make_cylinder()
    core.make_rock()
    core.compute_pore_size()
    core.set_water_saturation()

    hyd = HydrateFormation(core, formation_start=5, timestep=1.0, temperature=276.0, pressure=4.0,
                           salinity=3.5, morphology_type="grain_coating")
    states, mus = run(core, hyd, n_steps=60, E_keV=60, save_every=10)

    s = states[-1]
    print("shape", core.shape, "porosity", core.pore.sum() / core.inside.sum())
    print("S_w", s[IDX["brine"]].sum() / core.pore.sum(), "S_h", hyd.hydrate_saturation())
    print("fractions sum to 1:", np.allclose(s.sum(axis=0), 1, atol=1e-4))
    print("mu range [1/cm]", mus[-1].min(), mus[-1].max())