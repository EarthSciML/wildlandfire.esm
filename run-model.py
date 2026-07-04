#!/usr/bin/env python3
# run-model.py — minimal single-model runner for a 2-D EarthSci `.esm` (Python).
#
# The Python counterpart of run-model.jl / run-model-rs.sh. Same model, same
# `.esm`, same answer: a 2-D level-set Hamilton-Jacobi front  psi_t = -S |grad psi|
# on the doubly-periodic unit box, with |grad psi| discretized by the Godunov
# (Rouy-Tourin / Osher-Sethian) upwind scheme (imported by reference from the
# sibling earthscidiscretizations checkout on the model-ref mount edge — nothing
# is vendored here). psi starts as the signed distance to a circle, so the front
# {psi = 0} expands outward at speed S = R_0. Loads through the earthsci_toolkit
# Python binding, integrates with SciPy (LSODA) while sampling several snapshots,
# and plots the front location at each time as a single figure with time encoded
# by color (PNG). No conformance / MMS checks.
#
# Usage:
#   python run-model.py [model.esm] [t_end]
#     model.esm   path to the .esm to run   (default: ./wildlandfire.esm)
#     t_end       final integration time    (default: 2.0)
#
# Environment: needs the earthsci_toolkit binding (dev'd in the sibling
# EarthSciSerialization checkout — set ESS_ROOT to override) plus numpy / scipy,
# and matplotlib for the PNG (the run still prints radii without it). The
# companion run-model-py.sh provisions a minimal venv and invokes this script.

import os
import re
import sys
import math

HERE = os.path.dirname(os.path.abspath(__file__))
ESS_ROOT = os.environ.get(
    "ESS_ROOT", os.path.normpath(os.path.join(HERE, "..", "EarthSciSerialization"))
)
if not os.path.isfile(os.path.join(ESS_ROOT, "esm-schema.json")):
    sys.exit(
        f"EarthSciSerialization not found at '{ESS_ROOT}'; set ESS_ROOT or clone "
        "it as a sibling checkout of this repo."
    )
# Use the dev'd Python binding straight from source (no install step).
sys.path.insert(0, os.path.join(ESS_ROOT, "packages", "earthsci_toolkit", "src"))

import numpy as np
import earthsci_toolkit as esm

NT = 6      # number of front snapshots (t = 0 … t_end)
N = 32      # grid resolution (metaparameters NX = NY = N); matches the Julia and
            # Rust runners, capped here by the non-smooth Godunov RHS cost.
R0 = 0.1    # LevelSetFireSpread.R_0 override (m/s): the mounted component's
            # default R_0=1.0 was calibrated for its old 500 m domain; on the
            # unit box that overshoots, so we set the isotropic speed to 0.1
            # (front reaches 0.15 + 0.1·t_end, staying clear of the seam).

model_path = sys.argv[1] if len(sys.argv) >= 2 else \
    os.path.join(HERE, "wildlandfire.esm")
t_end = float(sys.argv[2]) if len(sys.argv) >= 3 else 2.0

print(f"loading    {model_path}  (NX=NY={N})")
file = esm.load(model_path, metaparameters={"NX": N, "NY": N})

print(f"simulating (0.0, {t_end}) with LSODA, {NT} snapshots …")
sim = esm.simulate(
    file, (0.0, t_end),
    parameters={"LevelSetFireSpread.R_0": R0},
    method="LSODA", rtol=1e-2, atol=1e-3,
)
if not sim.success:
    sys.exit(f"solver failed: {sim.message}")

# Assemble the primary 2-D field: element rows "<stem>[i,j]", grouped by stem,
# indices parsed as integers. sim.vars are the row labels of sim.y (n_rows × n_t).
groups = {}   # stem -> [(i, j, row)]
cell_re = re.compile(r"^(.*?)\[(\d+),\s*(\d+)\]$")
for row, name in enumerate(sim.vars):
    m = cell_re.match(name)
    if m is None:
        continue
    stem, i, j = m.group(1), int(m.group(2)), int(m.group(3))
    groups.setdefault(stem, []).append((i, j, row))
if not groups:
    sys.exit("no 2-D indexed state field found to plot")

# Prefer the level-set state field psi (the mounted component also exposes many
# [x,y] observeds); fall back to the widest 2-D field if no psi is present.
psi_stems = [k for k in groups if k.endswith("psi")]
stem = max(psi_stems or list(groups), key=lambda k: len(groups[k]))
cells = groups[stem]
NX = max(c[0] for c in cells)
NY = max(c[1] for c in cells)
xs = [(i - 0.5) / NX for i in range(1, NX + 1)]
ys = [(j - 0.5) / NY for j in range(1, NY + 1)]

# SciPy's simulate() samples a dense trajectory rather than exact save times, so
# reconstruct psi at the NT requested snapshots by linear interpolation of each
# element row (the fixture-runner convention; matches the Julia saveat output to
# the plot's precision). psi[k] is the NX×NY field at snapshot time ts[k].
t = np.asarray(sim.t, dtype=float)
Y = np.asarray(sim.y, dtype=float)
ts = list(np.linspace(0.0, t_end, NT))
psi = [np.full((NX, NY), np.nan) for _ in ts]
for (i, j, row) in cells:
    yi = np.interp(ts, t, Y[row])
    for k in range(len(ts)):
        psi[k][i - 1, j - 1] = yi[k]


def zero_segments(xs, ys, Z):
    """Marching squares for the zero contour of a field sampled at (xs[i], ys[j]);
    returns line segments packed into (X, Y) with NaN breaks (one plot series)."""
    X, Y = [], []

    def interp(a, b):
        s = a[2] / (a[2] - b[2])
        return (a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]))

    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            c = ((xs[i],     ys[j],     Z[i, j]),
                 (xs[i + 1], ys[j],     Z[i + 1, j]),
                 (xs[i + 1], ys[j + 1], Z[i + 1, j + 1]),
                 (xs[i],     ys[j + 1], Z[i, j + 1]))     # CCW
            pts = []
            for e in range(4):
                a = c[e]
                b = c[(e + 1) % 4]
                if (a[2] < 0) != (b[2] < 0):
                    pts.append(interp(a, b))
            if len(pts) >= 2:
                X += [pts[0][0], pts[1][0], np.nan]
                Y += [pts[0][1], pts[1][1], np.nan]
    return X, Y


cellarea = (1.0 / NX) * (1.0 / NY)
print(f"\nsolver: success; field '{stem}' is {NX}×{NY}")
for k, tt in enumerate(ts):
    burning = int(np.sum(psi[k] < 0))
    r = math.sqrt(burning * cellarea / math.pi)
    print(f"  t={tt:.2f}  front r_eff={r:.4f}")

# --- single figure: front (psi=0) at each time, color = time; PNG output -------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    print("\n(matplotlib not available — skipping PNG; radii above are the check)")
    sys.exit(0)

cg = LinearSegmentedColormap.from_list(
    "seq_blue", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])   # early → late
fig, ax = plt.subplots(figsize=(6.2, 5.4))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")
for k, tt in enumerate(ts):
    X, Y = zero_segments(xs, ys, psi[k])
    frac = 1.0 if len(ts) == 1 else k / (len(ts) - 1)
    ax.plot(X, Y, color=cg(frac), lw=2, label=f"{tt:.2f}")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"{os.path.basename(model_path)} — front {{{stem}=0}}, Python / LSODA",
             fontsize=11)
leg = ax.legend(title="t", fontsize=8, loc="upper left",
                bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
leg.get_title().set_fontsize(9)
for spine in ax.spines.values():
    spine.set_color("#52514e")
fig.tight_layout()

plot_path = os.path.join(os.path.dirname(os.path.abspath(model_path)),
                         "front_python.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"wrote plot: {plot_path}")
