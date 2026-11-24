"""
2D samples for CT simulation.

Each sample is a label image where every pixel holds an integer naming its material
(0 = vacuum). mu_map turns the labels into linear attenuation at any energy, so the
same sample works for both the monochromatic and the polychromatic beam scripts.
"""
import numpy as np
import scipy.stats as spst
from scipy import ndimage as ndi
import xraylib as xrl
import porespy as ps


# name -> (chemical formula, density [g/cm^3])
MATERIALS = {
    "water":     ("H2O",             1.000),
    "ice":       ("H2O",             0.917),
    "hydrate":   ("CO2(H2O)6.08",    1.09),
    "CO2 gas":   ("CO2",             0.07),
    "quartz":    ("SiO2",            2.65),
    "feldspar":  ("KAlSi3O8",        2.56),
    "kaolinite": ("Al2Si2O5(OH)4",   2.60),
    "bone":      ("Ca10(PO4)6(OH)2", 1.92),
    "iron":      ("Fe",              7.87),
    "aluminum":  ("Al",              2.70),
}

# Bentheimer grain composition, XRD weight fractions (Peksa et al. 2015, Table 1).
# Clay = kaolinite 2.50 + montmorillonite 0.18; the remaining 0.7 wt% (carbonates, oxides, pyrite) is left out.
SANDSTONE_WT = {"quartz": 0.9170, "feldspar": 0.0486, "kaolinite": 0.0268}
_vol = {m: w / MATERIALS[m][1] for m, w in SANDSTONE_WT.items()}      # weight / density = volume
SANDSTONE = {m: v / sum(_vol.values()) for m, v in _vol.items()}       # volume fractions, sum to 1

# Integer label of each material in the label images. 0 is vacuum.
NAMES = ["vacuum", *MATERIALS, "sandstone"]
LABEL = {name: i for i, name in enumerate(NAMES)}

# Hydrate volume formed per volume of water consumed, from the formula CO2·6.08 H2O and the densities above
M_CO2, M_H2O, N_HYD = 44.01, 18.015, 6.08
EXPANSION = ((M_CO2 + N_HYD * M_H2O) / MATERIALS["hydrate"][1]) / (N_HYD * M_H2O / MATERIALS["water"][1])


def density(name):
    """Density [g/cm^3]. Sandstone is the volume-weighted average of its minerals."""
    if name == "sandstone":
        return sum(f * MATERIALS[m][1] for m, f in SANDSTONE.items())
    return MATERIALS[name][1]


def mu(name, energy):
    """Linear attenuation [1/mm] at one energy or an array of energies [keV]."""
    if name == "vacuum":
        return np.zeros(np.shape(energy)) if np.ndim(energy) else 0.0
    if name == "sandstone":
        return sum(f * mu(m, energy) for m, f in SANDSTONE.items())
    formula, rho = MATERIALS[name]
    E = np.atleast_1d(np.asarray(energy, float))
    out = np.array([xrl.CS_Total_CP(formula, float(e)) * rho / 10 for e in E])   # [cm^2/g]*[g/cm^3] = [1/cm], /10 -> [1/mm]
    return out if np.ndim(energy) else out[0]


def mass_att(name, energy):
    """Mass attenuation coefficient mu/rho [cm^2/g]."""
    return mu(name, energy) * 10 / density(name)


class Sample2D:
    """
    A circular sample on an n_pixel x n_pixel grid.

    n_pixel    : image size in pixels
    pixel_size : mm per pixel
    radius     : sample radius in mm
    seed       : same seed gives the same sample every run
    """

    def __init__(self, n_pixel, pixel_size, radius, seed=0):
        self.n_pixel, self.pixel_size, self.radius, self.seed = n_pixel, pixel_size, radius, seed
        self.c = (n_pixel - 1) / 2                     # centre in pixels
        self.r_px = radius / pixel_size                # radius in pixels
        if self.r_px > n_pixel / 2:
            raise ValueError("The sample does not fit in the image")
        self.yy, self.xx = np.mgrid[:n_pixel, :n_pixel]
        self.disk = np.hypot(self.xx - self.c, self.yy - self.c) <= self.r_px
        self.labels = np.zeros((n_pixel, n_pixel), np.uint8)

    # ---------- basics ----------

    def make_disk(self, material):
        """Fill the disk with one material; everything outside is vacuum."""
        self.labels = np.where(self.disk, LABEL[material], LABEL["vacuum"]).astype(np.uint8)
        return self.labels

    def mu_map(self, energy):
        """
        Linear attenuation image [1/mm] from the labels.
        Scalar energy -> (n, n). Array of energies -> (n_E, n, n).
        """
        E = np.atleast_1d(np.asarray(energy, float))
        out = np.zeros((E.size, *self.labels.shape))
        for lab in np.unique(self.labels):
            if lab != LABEL["vacuum"]:
                out[:, self.labels == lab] = np.atleast_1d(mu(NAMES[lab], E))[:, None]
        return out if np.ndim(energy) else out[0]

    def masks(self):
        """One boolean mask per material present (vacuum left out). Used for per-material forward projection."""
        return {NAMES[lab]: self.labels == lab for lab in np.unique(self.labels) if lab != LABEL["vacuum"]}

    # ---------- static samples ----------

    def simple_sample(self, inclusions, container="aluminum"):
        """
        Disk of `container` with smaller disks inside it.
        inclusions : list of (material, radius in mm), placed at random without overlapping.
        Centres in mm are stored in self.centres_mm (same coordinates as extent=(0, L, L, 0)).
        """
        rng = np.random.default_rng(self.seed)
        self.make_disk(container)
        c, R = self.c, self.r_px
        placed = []                                     # (x, y, r) in pixels
        for material, r_mm in inclusions:
            r = r_mm / self.pixel_size
            for _ in range(1000):
                x, y = rng.uniform(c - (R - r), c + (R - r), size=2)
                inside = np.hypot(x - c, y - c) <= R - r - 1
                free = all(np.hypot(x - xj, y - yj) >= r + rj + 1 for xj, yj, rj in placed)
                if inside and free:
                    break
            else:
                raise RuntimeError("Could not place all inclusions; try smaller or fewer")
            placed.append((x, y, r))
            self.labels[(self.xx - x) ** 2 + (self.yy - y) ** 2 <= r ** 2] = LABEL[material]
        self.centres_mm = [((x + 0.5) * self.pixel_size, (y + 0.5) * self.pixel_size) for x, y, _ in placed]
        return self.labels

    def rock_sample(self, porosity, grain_radius, pore_fluid="water"):
        """
        Sandstone grains (PoreSpy overlapping disks, radius 0.7-1.3 x grain_radius [mm]) with pores filled by pore_fluid.
        Also stores self.grain, self.pore and self.fill_order for hydrate_growth.
        """
        # PoreSpy's porosity is not exact for a spread of radii, so adjust the request until the disk matches
        r_px = grain_radius / self.pixel_size
        radii = spst.uniform(loc=0.7 * r_px, scale=0.6 * r_px)
        target = porosity
        for _ in range(5):
            void = ps.generators.polydisperse_spheres(shape=[self.n_pixel, self.n_pixel], porosity=target,
                                                      dist=radii, r_min=int(0.7 * r_px), seed=self.seed)
            measured = void[self.disk].mean()
            if abs(measured - porosity) < 0.005:
                break
            target += porosity - measured
        self.grain = ~void & self.disk
        self.pore = void & self.disk

        # CO2 fill order (capillary drainage). In a water-wet rock CO2 enters wide pores first because it needs
        # less pressure there. Local pore width = radius of the largest disk that fits in the pore and covers the pixel.
        size = ps.filters.local_thickness(self.pore)
        d_wall = ndi.distance_transform_edt(self.pore)
        order = np.flatnonzero(self.pore)
        self.fill_order = order[np.lexsort((-d_wall.flat[order], -size.flat[order]))]   # widest pores first, centres first

        self.make_disk("sandstone")
        self.labels[self.pore] = LABEL[pore_fluid]
        return self.labels

    # ---------- dynamic sample ----------

    def hydrate_growth(self, S_H, S_CO2, n_nuclei=20, lateral=5.0):
        """
        Put CO2, water and hydrate into the pores of the rock from rock_sample.
        S_H, S_CO2 : final saturations (fractions of pore volume)
        n_nuclei   : number of hydrate nuclei on the CO2-water interface
        lateral    : how much faster hydrate spreads along the interface than it thickens
        """
        if not hasattr(self, "fill_order"):
            raise RuntimeError("Run rock_sample first")
        if S_H + S_CO2 > 1 or min(S_H, S_CO2) < 0:
            raise ValueError("Need S_H, S_CO2 >= 0 and S_H + S_CO2 <= 1")
        WATER, CO2, HYDRATE = LABEL["water"], LABEL["CO2 gas"], LABEL["hydrate"]
        rng = np.random.default_rng(self.seed)
        order = self.fill_order

        # Pixel count of each phase. n_wat0 is the water before hydrate formed.
        S_w = 1 - S_H - S_CO2
        n_hyd, n_wat = round(S_H * order.size), round(S_w * order.size)
        n_wat0 = n_wat + round(n_hyd / EXPANSION)
        if n_wat0 > order.size:
            raise ValueError("Too much hydrate for the pore volume")

        # CO2 in the widest pores, water in the rest
        labels = np.where(self.disk, LABEL["sandstone"], LABEL["vacuum"]).astype(np.uint8)
        labels.flat[order[:order.size - n_wat0]] = CO2
        labels.flat[order[order.size - n_wat0:]] = WATER

        # Hydrate growth (own model). Arrival time = distance from the interface (thickening)
        # + distance from the nearest nucleus / lateral (fast spreading along the interface).
        if n_hyd > 0:
            water, co2 = labels == WATER, labels == CO2
            d_int = np.where(water, ndi.distance_transform_edt(~co2), ndi.distance_transform_edt(~water))
            interface = np.flatnonzero(water & ndi.binary_dilation(co2))
            not_nucleus = np.ones(labels.shape, bool)
            not_nucleus.flat[rng.choice(interface, min(n_nuclei, interface.size), replace=False)] = False
            t = d_int + ndi.distance_transform_edt(not_nucleus) / lateral
            # Hydrate replaces the converted water, and the extra volume (expansion) grows into the CO2
            for phase, n in ((water, n_wat0 - n_wat), (co2, n_hyd - (n_wat0 - n_wat))):
                p = np.flatnonzero(phase)
                labels.flat[p[np.argsort(t.flat[p], kind="stable")[:n]]] = HYDRATE

        self.labels = labels
        return labels

    def growth_stages(self, S_H, S_CO2, **kwargs):
        """The three stages up to the final sample, as a list of (labels, title). Ends with self.labels = final stage."""
        S_w = 1 - S_H - S_CO2
        S_CO2_before = 1 - (S_w + S_H / EXPANSION)          # CO2 saturation before hydrate formed
        rock = np.where(self.disk, LABEL["sandstone"], LABEL["vacuum"]).astype(np.uint8)
        rock[self.pore] = LABEL["water"]
        return [
            (rock, "1. Rock, pores filled with water"),
            (self.hydrate_growth(0.0, S_CO2_before).copy(), f"2. After CO$_2$ injection, $S_{{CO_2}}$={S_CO2_before:.2f}"),
            (self.hydrate_growth(S_H, S_CO2, **kwargs), f"3. After hydrate formation, $S_H$={S_H}, $S_{{CO_2}}$={S_CO2}"),
        ]

    # ---------- checks ----------

    def porosity(self):
        return self.pore.sum() / self.disk.sum()

    def saturations(self):
        """Fraction of pore volume held by each fluid phase."""
        n = self.pore.sum()
        return {name: (self.labels == LABEL[name]).sum() / n for name in ("water", "hydrate", "CO2 gas")}