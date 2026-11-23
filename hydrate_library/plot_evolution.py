import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from Hydrate_core import IDX, PHASES

def plot_evolution(core, states, mus, save_every=1, timestep=1.0, axis=1):
    "Middle slice of each saved step: dominant phase, hydrate fraction and mu"
    mid = core.shape[axis] // 2
    take = lambda a: np.take(a, mid, axis=axis)           # axis=1 or 2: vertical slice, axis=0: horizontal

    colors = {"quartz": "#d9c9a3", "feldspar": "#c98f6b", "kaolinite": "#a7a7a7", "brine": "#3a7bd5",
              "fluid": "#f2f2f2", "hydrate": "#e41a1c", "ice": "#9be7ff", "outside": "#000000"}
    extra = ["#8c564b", "#bcbd22", "#9467bd", "#17becf", "#e377c2", "#7f7f7f"]   # for minerals added in the config
    for p in PHASES:
        if p not in colors:
            colors[p] = extra.pop(0) if extra else "#7f6a4d"
    cmap = ListedColormap([colors[p] for p in PHASES])

    n = len(states)
    fig, ax = plt.subplots(3, n, figsize=(2.2 * n, 7), squeeze=False)
    hmax = max(s[IDX["hydrate"]].max() for s in states) or 1
    vmin, vmax = min(m.min() for m in mus), max(m.max() for m in mus)

    for j, (s, m) in enumerate(zip(states, mus)):
        t = j * save_every * timestep
        ax[0, j].imshow(take(s.argmax(axis=0)), cmap=cmap, vmin=0, vmax=len(PHASES) - 1,
                        origin="lower", interpolation="nearest")
        im_h = ax[1, j].imshow(take(s[IDX["hydrate"]]), cmap="Reds", vmin=0, vmax=hmax, origin="lower")
        im_m = ax[2, j].imshow(take(m), cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        ax[0, j].set_title(f"t = {t:g} s")
        for a in ax[:, j]:
            a.set_xticks([]); a.set_yticks([])

    ax[0, 0].set_ylabel("dominant phase"); ax[1, 0].set_ylabel("hydrate fraction"); ax[2, 0].set_ylabel("mu")
    fig.colorbar(im_h, ax=ax[1, :], shrink=0.8)
    fig.colorbar(im_m, ax=ax[2, :], shrink=0.8, label="1/cm")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[p]) for p in PHASES if p != "ice"]
    fig.legend(handles, [p for p in PHASES if p != "ice"], loc="upper center", ncol=7, frameon=False)
    return fig

def plot_saturations(core, states, save_every=1, timestep=1.0):
    "Brine, CO2 and hydrate saturation of the whole sample over time"
    t = np.arange(len(states)) * save_every * timestep
    V_pore = core.pore.sum()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for p, c in [("brine", "#3a7bd5"), ("fluid", "#888888"), ("hydrate", "#e41a1c")]:
        ax.plot(t, [s[IDX[p]].sum() / V_pore for s in states], "o-", color=c, label=p)
    ax.set_xlabel("time [s]"); ax.set_ylabel("saturation"); ax.legend()
    fig.tight_layout()
    return fig