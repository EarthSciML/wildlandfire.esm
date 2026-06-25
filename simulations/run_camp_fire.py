#!/usr/bin/env python3
"""Run the Camp Fire level-set simulation through the EarthSciSerialization
Python pipeline — the runnable successor to the legacy WildlandFire.jl script.

`camp_fire.esm` is the *declarative* coupled-system spec (interface stubs +
coupling). This script *instantiates* it with the authoritative EarthSciModels
components and drives them end-to-end:

    flatten            (earthsci_toolkit.flatten)        — resolve param_to_var coupling
    flattened_to_esm   (earthsci_toolkit.spatial_discretize)
    spatial_discretize (… , GDD)                         — lower grad/|grad psi| via catalog rules
    simulate           (earthsci_toolkit.simulation)     — integrate the ArrayOp ODE system

Two runs:
  1. standalone LevelSetFireSpread  — eikonal front at R_0 (no coupling);
  2. RothermelFireSpread -> LevelSetFireSpread — the fire-front driven by the
     Rothermel-computed rate of spread (the fire-behavior chain wired in).

Data-driven inputs (LANDFIRE fuel codes, USGS3DEP terrain, ERA5 wind) are held
at constants here — the full data-loader path needs live data at runtime.

Requirements (no live network/data needed):
  - earthsci_toolkit on PYTHONPATH (EarthSciSerialization/packages/earthsci_toolkit/src)
  - EarthSciModels checked out (set $EARTHSCIMODELS, default ../EarthSciModels)
  - numpy, scipy

The grid is coarsened from the component's native 5 m (101x101) to keep the
demo fast; raise the resolution by lowering GRID_SPACING.
"""
from __future__ import annotations

import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")  # quiet the 0.5.0-vs-0.4.0 toolkit version notice

import earthsci_toolkit as et
from earthsci_toolkit.simulation import simulate
from earthsci_toolkit.spatial_discretize import spatial_discretize, flattened_to_esm

EARTHSCIMODELS = Path(os.environ.get("EARTHSCIMODELS", Path(__file__).resolve().parents[2] / "EarthSciModels"))
WILDLAND = EARTHSCIMODELS / "components" / "wildland_fire"
GRID_SPACING = 50.0          # m (native is 5.0 -> 101x101; coarsened for speed)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _godunov_gdd() -> dict:
    """GDD selecting: Godunov upwind for |grad psi| (stable level-set front) and
    centered 2nd-order for the individual grad(psi, .) in the wind/slope projections."""
    def ix(ox, oy):
        return {"op": "index", "args": ["$u", "$x" if ox == 0 else {"op": "+", "args": ["$x", ox]},
                                        "$y" if oy == 0 else {"op": "+", "args": ["$y", oy]}]}

    def comp(m, c, p):
        dm = {"op": "/", "args": [{"op": "-", "args": [c, m]}, "dx"]}
        dp = {"op": "/", "args": [{"op": "-", "args": [p, c]}, "dx"]}
        return {"op": "+", "args": [{"op": "^", "args": [{"op": "max", "args": [dm, 0]}, 2]},
                                    {"op": "^", "args": [{"op": "min", "args": [dp, 0]}, 2]}]}
    c = ix(0, 0)
    return {"discretizations": {
        "grad_norm": {
            "applies_to": {"op": "sqrt", "args": [{"op": "+", "args": [
                {"op": "^", "args": [{"op": "grad", "args": ["$u"], "dim": "$x"}, 2]},
                {"op": "^", "args": [{"op": "grad", "args": ["$u"], "dim": "$y"}, 2]}]}]},
            "grid_family": "cartesian",
            "replacement": {"op": "sqrt", "args": [{"op": "+", "args": [
                comp(ix(-1, 0), c, ix(1, 0)), comp(ix(0, -1), c, ix(0, 1))]}]}},
        "grad": {
            "applies_to": {"op": "grad", "args": ["$u"], "dim": "$x"},
            "grid_family": "cartesian",
            "replacement": {"op": "arrayop", "output_idx": ["$x"], "args": ["$u"], "expr":
                {"op": "/", "args": [{"op": "-", "args": [
                    {"op": "index", "args": ["$u", {"op": "+", "args": ["$x", 1]}]},
                    {"op": "index", "args": ["$u", {"op": "+", "args": ["$x", -1]}]}]},
                    {"op": "*", "args": [2, "dx"]}]}}}}}


def _coarsen(domains: dict) -> dict:
    for ax in ("x", "y"):
        domains["fire_domain"]["spatial"][ax]["grid_spacing"] = GRID_SPACING
    return domains


def _front_radius(r, state, n, X, t, center):
    jc = (n + 1) // 2
    xs = [X(i) for i in range(jc, n + 1)]
    vs = [float(np.interp(t, r.t, r.y[next(k for k, nm in enumerate(r.vars)
                                           if nm.endswith(f"[{i},{jc}]"))]))
          for i in range(jc, n + 1)]
    for k in range(len(vs) - 1):
        if vs[k] <= 0 <= vs[k + 1]:
            return xs[k] - center + (xs[k + 1] - xs[k]) * (-vs[k]) / (vs[k + 1] - vs[k])
    return float("nan")


def _run(esm_or_disc, label, *, r0=100.0, center=250.0, tspan=(0.0, 40.0)):
    disc = esm_or_disc
    state = next(n for n, v in next(iter(disc["models"].values()))["variables"].items()
                 if v["type"] == "state")
    f = et.load(disc)
    sp = GRID_SPACING
    n = int(round(500.0 / sp)) + 1

    def X(i):
        return (i - 1) * sp

    ic = {f"{state}[{i},{j}]": math.hypot(X(i) - center, X(j) - center) - r0
          for i in range(1, n + 1) for j in range(1, n + 1)}
    r = simulate(f, tspan, initial_conditions=ic, method="LSODA", rtol=1e-5, atol=1e-7)
    if not r.success:
        print(f"  [{label}] FAILED: {r.message}")
        return
    print(f"  [{label}] solved ({len(r.t)} steps). Fire-front radius from ignition:")
    for t in np.linspace(tspan[0], tspan[1], 5):
        print(f"     t = {t:5.1f} s   r = {_front_radius(r, state, n, X, t, center):7.2f} m")


def run_standalone_level_set():
    """Real LevelSetFireSpread alone: eikonal front at R_0 (defaults: no wind/slope)."""
    ls = _load(WILDLAND / "level_set_fire_spread.esm")
    ls["esm"] = "0.5.0"
    _coarsen(ls["domains"])
    esm = {"esm": "0.5.0", "metadata": {"name": "LevelSetOnly"},
           "domains": ls["domains"], "models": {"LevelSetFireSpread": ls["models"]["LevelSetFireSpread"]}}
    disc = spatial_discretize(et_flatten_to_dict(esm, ls), _godunov_gdd())
    _run(disc, "standalone level-set (R_0=1 m/s)")


def run_coupled_rothermel_levelset():
    """Real RothermelFireSpread -> real LevelSetFireSpread via param_to_var.
    The 0-D Rothermel algebraic system computes the rate of spread from the
    (here-constant) fuel/wind/slope inputs; it drives the 2-D front."""
    ls = _load(WILDLAND / "level_set_fire_spread.esm")
    roth = _load(WILDLAND / "rothermel" / "fire_spread.esm")
    Rm = roth["models"]["RothermelFireSpread"]
    LSm = ls["models"]["LevelSetFireSpread"]
    _coarsen(ls["domains"])
    # Anderson FM1 (short grass) fuel bed + midflame wind 3 m/s + mild slope.
    for k, v in {"sigma": 11483.0, "w0": 0.166, "delta": 0.305, "Mx": 0.12,
                 "h": 18608000.0, "Mf": 0.08, "U": 3.0, "tan_phi": 0.1}.items():
        if k in Rm["variables"]:
            Rm["variables"][k]["default"] = v
    coupled = {
        "esm": "0.5.0", "metadata": {"name": "CampFireCoupled"}, "domains": ls["domains"],
        "models": {"RothermelFireSpread": Rm, "LevelSetFireSpread": LSm},
        "coupling": [{"type": "variable_map", "from": f"RothermelFireSpread.{a}",
                      "to": f"LevelSetFireSpread.{b}", "transform": "param_to_var",
                      "lifting": "pointwise"}
                     for a, b in [("R", "R_0"), ("C_coeff", "C_wind"), ("B_coeff", "B_wind"),
                                  ("E_coeff", "E_wind"), ("beta_ratio", "beta_ratio"),
                                  ("phi_s_coeff", "phi_s_coeff")]],
    }
    flat = et.flatten(et.load(coupled))
    esm = flattened_to_esm(flat, ls["domains"],
                           boundary_conditions=LSm["boundary_conditions"])
    disc = spatial_discretize(esm, _godunov_gdd())
    _run(disc, "coupled Rothermel -> level-set")


def et_flatten_to_dict(esm: dict, ls: dict) -> dict:
    """Single-component path: flatten (no coupling) then adapt, carrying BCs."""
    flat = et.flatten(et.load(esm))
    return flattened_to_esm(flat, ls["domains"],
                            boundary_conditions=ls["models"]["LevelSetFireSpread"]["boundary_conditions"])


def main() -> int:
    if not WILDLAND.is_dir():
        print(f"EarthSciModels components not found at {WILDLAND}; set $EARTHSCIMODELS.",
              file=sys.stderr)
        return 1
    print(f"Camp Fire level-set via the ESS Python pipeline (dx={GRID_SPACING} m)\n")
    run_standalone_level_set()
    run_coupled_rothermel_levelset()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
