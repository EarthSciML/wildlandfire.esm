#!/usr/bin/env python3
"""Camp Fire end-to-end — a THIN driver over the real ESS / ESD / ESIO pipeline.

This is the "no workarounds" successor to the staged driver. It does exactly
three things, on the FULL coupled ``camp_fire.esm`` (11 components, 20 loaders):

    load        et.load(camp_fire.esm) + et.flatten     — resolve the 28 coupling edges
    discretize  flattened_to_esm + spatial_discretize   — lower the level-set PDE
                                                           (ESD Godunov |grad psi| rule)
    run         simulate(..., provider_factory=esio)    — integrate; loaders served by
                                                           real EarthSciIO providers

Removed relative to the old staged driver (all numerical workarounds):
  * the precomputed constant rate-of-spread (``rate_of_spread``) — R is now
    computed per cell by the inlined Rothermel chain, the real coupling;
  * the level-set-only solve with R held constant (``level_set_front``);
  * the uniform representative LANDFIRE/USGS/ERA5 stand-ins (``REP``) and the
    hand-written numeric ignition IC — the domain's declared ``expression`` IC
    seeds the front (gap B2), and the loaders fetch real data via esio providers.

The integration window is SHORT by design: gap P (the coupled PDE inlines the
0-D fire physics per cell per timestep) makes the real 16 h window intractable in
the pure-Python interpreter. The goal is to drive the real pipeline as far as it
goes and surface remaining feasibility gaps — not to reproduce the fire.

Usage::

    PYTHONPATH=<ESS>/packages/earthsci_toolkit/src:<EarthSciIO> \
    EARTHSCIMODELS=<EarthSciModels> \
        python simulations/run_camp_fire_e2e_pde.py [--seconds 30]
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CAMP_FIRE_ESM = REPO / "simulations" / "camp_fire.esm"


def _godunov_gdd() -> dict:
    """The ESD Godunov ``|grad psi|`` discretization rule.

    Borrowed verbatim from ``run_camp_fire.py`` (the proven catalog rule that
    ``spatial_discretize`` consumes); kept in one place rather than duplicated.
    """
    spec = importlib.util.spec_from_file_location(
        "rcf", str(REPO / "simulations" / "run_camp_fire.py")
    )
    rcf = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(rcf)
    return rcf._godunov_gdd()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--seconds", type=float, default=30.0,
        help="integration window in seconds (short by design — gap P)",
    )
    args = ap.parse_args(argv)

    import earthsci_toolkit as et
    from earthsci_toolkit.simulation import simulate
    from earthsci_toolkit.spatial_discretize import flattened_to_esm, spatial_discretize
    from earthsci_toolkit.data_loaders.esio_provider import esio_provider_factory

    print("Camp Fire — thin driver (load camp_fire.esm -> discretize via ESD -> run via ESS)\n")

    # 1. LOAD — resolve the declarative coupled system into one flat system.
    raw = json.loads(CAMP_FIRE_ESM.read_text())
    flat = et.flatten(et.load(str(CAMP_FIRE_ESM)))
    d = flat.domain
    sx, sy = d.spatial["x"], d.spatial["y"]
    nx = int(round((sx.max - sx.min) / sx.grid_spacing)) + 1
    ny = int(round((sy.max - sy.min) / sy.grid_spacing)) + 1
    print("== load ==")
    print(f"   {len(flat.metadata.source_systems)} components, {len(flat.equations)} equations, "
          f"{len(flat.state_variables)} states, {len(flat.loader_fields)} loader fields")
    print(f"   domain {nx}x{ny} @ {sx.grid_spacing:g} m  "
          f"({d.temporal.start} -> {d.temporal.end})")

    # 2. DISCRETIZE — lower the level-set PDE with the ESD Godunov rule. The full
    #    coupled system (incl. the inlined 0-D fire behavior) becomes an ArrayOp
    #    ODE; the 20 loader fields are carried through (gap A).
    esm = flattened_to_esm(flat, raw["domains"], boundary_conditions=None)
    disc = spatial_discretize(esm, _godunov_gdd())
    f = et.load(disc)
    flat2 = et.flatten(f)
    print("== discretize (ESD Godunov |grad psi|) ==")
    print(f"   -> {len(flat2.state_variables)} ODE states, "
          f"{len(flat2.loader_fields)} loaders preserved")

    # 3. RUN — integrate; loaders resolve to real EarthSciIO providers (GeoTIFF /
    #    CDS readers + content-addressed cache) via the opt-in esio adapter. The
    #    declared expression IC seeds the front (gap B2). Window is short (gap P).
    print(f"== run (ESS simulate; loaders via real EarthSciIO providers); tspan=(0, {args.seconds:g} s) ==")
    t0 = time.time()
    r = simulate(
        f, (0.0, args.seconds), method="LSODA", rtol=1e-3, atol=1e-5,
        provider_factory=esio_provider_factory,
    )
    dt = time.time() - t0
    if not r.success:
        print(f"   simulate did not complete ({dt:.0f}s): {r.message}")
        return 1
    print(f"   solved in {dt:.0f}s: {len(r.t)} steps over {args.seconds:g} s, "
          f"{len(r.vars)} state elements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
