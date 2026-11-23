import yaml
from Hydrate_core import CoreSample, HydrateFormation, run, setup_materials, IDX
import matplotlib.pyplot as plt
from plot_evolution import plot_evolution, plot_saturations
def load_config(path="configs.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

cfg = load_config("configs.yaml")

core_cfg = dict(cfg["core"])
per_grain = core_cfg.pop("per_grain")      # not an __init__ argument, so take it out first

core = CoreSample(**core_cfg)
core.make_cylinder()
core.make_rock(per_grain=per_grain)
core.compute_pore_size()
core.set_water_saturation()

hyd = HydrateFormation(core, **cfg["hydrate"])
states, mus = run(core, hyd, **cfg["simulation"])

s = states[-1]
print("injected", core.injected_mm3, "in hydrate", hyd.co2_consumed_mm3,
      "free gas", s[IDX["fluid"]].sum() * core.voxel_volume_mm3)


sim = cfg["simulation"]
dt = cfg["hydrate"]["timestep"]
plot_evolution(core, states, mus, save_every=sim["save_every"], timestep=dt)
plot_saturations(core, states, save_every=sim["save_every"], timestep=dt)
plt.show()