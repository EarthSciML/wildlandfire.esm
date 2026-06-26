"""Data model for the Camp Fire validation harness.

Two concrete inputs flow into the harness:

* :class:`LevelSetRun` — the *simulated* fire progression. A signed level-set
  field ``psi(t, y, x)`` on a projected grid (Lambert Conformal Conic metres for
  the Camp Fire domain). The convention follows the level-set components: the
  burning/burned region is ``psi <= 0`` and the fire front is the ``psi = 0``
  contour. This is exactly what ``simulate()`` integrates in
  ``simulations/run_camp_fire.py``; :meth:`LevelSetRun.from_simulate_result`
  reconstructs the gridded field from that runner's flat ``state[i,j]`` output so
  the eventual live-data run (campaign bead E3) can pipe its result straight in.

* :class:`ObservedReference` — the *observed* Camp Fire (campaign phase E1):
  a final burned footprint (MTBS + last perimeter), an optional time series of
  daily burned-fraction grids (NIFC/GeoMAC perimeters), and an optional ignition
  point/time (VIIRS first detection). The E1 loaders rasterise every source to a
  ``burned_fraction(time, y, x)`` grid, so the reference reduces to plain arrays
  here — no GIS dependency, just numpy.

Both carry their own ``x``/``y`` coordinate vectors (projected metres). The
harness aligns them with :meth:`ObservedReference.regrid_to` when the observed
reference was rasterised onto a grid other than the run's.
"""
from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# Burned region of a signed level-set field: psi at or below this value.
DEFAULT_LEVELSET_THRESHOLD = 0.0
# Observed burned fraction at or above this value counts a cell as burned.
DEFAULT_FRACTION_THRESHOLD = 0.5

_VAR_INDEX_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*$")


def _uniform_spacing(coord: np.ndarray, axis_name: str) -> float:
    """Return the (uniform) spacing of a monotone coordinate vector.

    Areas and distances are reported in projected metres, which only makes
    sense on a regular grid; a non-uniform axis is a setup error worth catching
    early rather than silently mis-reporting square kilometres.
    """
    coord = np.asarray(coord, dtype=float)
    if coord.size < 2:
        raise ValueError(f"{axis_name} axis needs >= 2 points, got {coord.size}")
    diffs = np.diff(coord)
    step = float(np.mean(diffs))
    if step == 0 or not np.allclose(diffs, step, rtol=1e-4, atol=abs(step) * 1e-6):
        raise ValueError(f"{axis_name} axis must be uniformly spaced; got diffs {diffs!r}")
    return step


def cell_area(x: np.ndarray, y: np.ndarray) -> float:
    """Area of one grid cell in m^2 for uniformly spaced ``x``/``y`` (metres)."""
    return abs(_uniform_spacing(x, "x")) * abs(_uniform_spacing(y, "y"))


def mesh(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(Xg, Yg)`` coordinate grids of shape ``(ny, nx)`` (row = y, col = x)."""
    return np.meshgrid(np.asarray(x, float), np.asarray(y, float), indexing="xy")


def _to_datetime(value) -> Optional[_dt.datetime]:
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=_dt.timezone.utc)
    if isinstance(value, str):
        text = value.replace("Z", "+00:00")
        parsed = _dt.datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=_dt.timezone.utc)
    raise TypeError(f"cannot interpret {value!r} as a datetime")


def _iso(value: Optional[_dt.datetime]) -> Optional[str]:
    return None if value is None else value.astimezone(_dt.timezone.utc).isoformat()


@dataclass
class LevelSetRun:
    """A simulated level-set fire run: ``psi(t, y, x)`` on a projected grid.

    Parameters
    ----------
    psi:
        Signed level-set field, shape ``(n_t, n_y, n_x)``. Burned/burning where
        ``psi <= threshold`` (default 0); the front is the zero contour.
    times:
        Elapsed seconds since ``t0`` for each time slice, shape ``(n_t,)``.
    x, y:
        Projected coordinate vectors (metres), lengths ``n_x`` / ``n_y``.
    t0:
        Ignition / window-start instant, used to align ``times`` with the
        observed reference's absolute timestamps. Optional.
    """

    psi: np.ndarray
    times: np.ndarray
    x: np.ndarray
    y: np.ndarray
    t0: Optional[_dt.datetime] = None

    def __post_init__(self) -> None:
        self.psi = np.asarray(self.psi, dtype=float)
        self.times = np.asarray(self.times, dtype=float)
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        self.t0 = _to_datetime(self.t0)
        if self.psi.ndim != 3:
            raise ValueError(f"psi must be 3-D (t, y, x); got shape {self.psi.shape}")
        n_t, n_y, n_x = self.psi.shape
        if self.times.shape != (n_t,):
            raise ValueError(f"times {self.times.shape} != ({n_t},)")
        if self.y.shape != (n_y,) or self.x.shape != (n_x,):
            raise ValueError(
                f"coord/grid mismatch: psi {self.psi.shape}, "
                f"y {self.y.shape}, x {self.x.shape}"
            )
        if np.any(np.diff(self.times) < 0):
            raise ValueError("times must be non-decreasing")

    # -- geometry ---------------------------------------------------------
    @property
    def cell_area(self) -> float:
        return cell_area(self.x, self.y)

    @property
    def n_times(self) -> int:
        return self.psi.shape[0]

    def burned_mask(self, t_index: int, threshold: float = DEFAULT_LEVELSET_THRESHOLD) -> np.ndarray:
        """Boolean burned mask at a stored time index."""
        return self.psi[t_index] <= threshold

    def field_at(self, elapsed_s: float) -> np.ndarray:
        """Level-set field linearly interpolated to ``elapsed_s`` seconds.

        Clamped to the run window at both ends so callers can probe an observed
        perimeter time that sits slightly outside the saved range.
        """
        t = self.times
        if elapsed_s <= t[0]:
            return self.psi[0]
        if elapsed_s >= t[-1]:
            return self.psi[-1]
        hi = int(np.searchsorted(t, elapsed_s, side="right"))
        lo = hi - 1
        span = t[hi] - t[lo]
        w = 0.0 if span == 0 else (elapsed_s - t[lo]) / span
        return (1.0 - w) * self.psi[lo] + w * self.psi[hi]

    def mask_at(self, elapsed_s: float, threshold: float = DEFAULT_LEVELSET_THRESHOLD) -> np.ndarray:
        return self.field_at(elapsed_s) <= threshold

    def datetime_at(self, elapsed_s: float) -> Optional[_dt.datetime]:
        if self.t0 is None:
            return None
        return self.t0 + _dt.timedelta(seconds=float(elapsed_s))

    def elapsed_for(self, when: _dt.datetime) -> Optional[float]:
        """Seconds since ``t0`` for an absolute instant (``None`` if no ``t0``)."""
        if self.t0 is None:
            return None
        return (_to_datetime(when) - self.t0).total_seconds()

    # -- IO ---------------------------------------------------------------
    def save_npz(self, path: str | Path) -> Path:
        """Persist as a compressed ``.npz`` — the interchange E3 hands the harness."""
        path = Path(path)
        np.savez_compressed(
            path,
            psi=self.psi,
            times=self.times,
            x=self.x,
            y=self.y,
            t0=np.array(_iso(self.t0) or "", dtype=object),
        )
        return path if path.suffix else path.with_suffix(".npz")

    @classmethod
    def load_npz(cls, path: str | Path) -> "LevelSetRun":
        with np.load(path, allow_pickle=True) as data:
            t0 = str(data["t0"]) if "t0" in data else ""
            return cls(
                psi=data["psi"],
                times=data["times"],
                x=data["x"],
                y=data["y"],
                t0=t0 or None,
            )

    @classmethod
    def from_simulate_result(
        cls,
        result,
        *,
        x: Optional[Sequence[float]] = None,
        y: Optional[Sequence[float]] = None,
        dx: Optional[float] = None,
        origin: tuple[float, float] = (0.0, 0.0),
        state: Optional[str] = None,
        t0=None,
    ) -> "LevelSetRun":
        """Build a run from an EarthSciSerialization ``simulate()`` result.

        The runner integrates a flattened system whose variables are named
        ``"<state>[i,j]"`` with 1-based ``i`` (x index) and ``j`` (y index) — see
        ``run_camp_fire.py``. ``result`` only needs the duck-typed attributes
        ``t`` (``(n_t,)``), ``y`` (``(n_state, n_t)``) and ``vars`` (the matching
        variable names), so a real ``OdeResult`` or a light stand-in both work.

        Provide grid coordinates either explicitly (``x``/``y``) or via uniform
        ``dx`` + ``origin``; ``state`` selects the variable family when several
        share the array (defaults to the family carrying ``[i,j]`` indices).
        """
        names = list(result.vars)
        values = np.asarray(result.y, dtype=float)
        times = np.asarray(result.t, dtype=float)
        if values.shape[0] != len(names):
            raise ValueError(
                f"result.y has {values.shape[0]} rows but {len(names)} variable names"
            )

        parsed: list[tuple[int, int, int]] = []  # (row_index, i, j)
        for row, name in enumerate(names):
            m = _VAR_INDEX_RE.search(name)
            if not m:
                continue
            base = name[: m.start()].rstrip()
            base = base.split(".")[-1]
            if state is not None and base != state:
                continue
            parsed.append((row, int(m.group(1)), int(m.group(2))))
        if not parsed:
            raise ValueError("no '<state>[i,j]' grid variables found in result.vars")

        i_max = max(p[1] for p in parsed)
        j_max = max(p[2] for p in parsed)
        n_t = times.shape[0]
        psi = np.full((n_t, j_max, i_max), np.nan, dtype=float)
        for row, i, j in parsed:
            psi[:, j - 1, i - 1] = values[row]
        if np.isnan(psi).any():
            missing = int(np.isnan(psi[0]).sum())
            raise ValueError(f"reconstructed grid has {missing} unfilled cells (ragged index set)")

        if x is None or y is None:
            if dx is None:
                raise ValueError("provide x/y coordinate vectors or a uniform dx")
            x = origin[0] + dx * np.arange(i_max)
            y = origin[1] + dx * np.arange(j_max)
        return cls(psi=psi, times=times, x=x, y=y, t0=t0)


@dataclass
class ObservedReference:
    """Observed Camp Fire reference data (campaign phase E1).

    Parameters
    ----------
    burned_fraction_final:
        Final burned footprint as a fraction-of-cell-burned grid ``(n_y, n_x)``
        in ``[0, 1]`` — MTBS severity footprint unioned with the last perimeter.
    x, y:
        Projected coordinate vectors (metres) for the reference grid.
    perimeter_times, burned_fraction_series:
        Optional daily progression: timestamps and matching
        ``(n_k, n_y, n_x)`` burned-fraction grids (NIFC/GeoMAC perimeters).
    ignition_xy, ignition_time:
        Optional ignition point ``(x, y)`` in projected metres and instant
        (VIIRS/MODIS first active-fire detection).
    """

    burned_fraction_final: np.ndarray
    x: np.ndarray
    y: np.ndarray
    perimeter_times: Optional[list] = None
    burned_fraction_series: Optional[np.ndarray] = None
    ignition_xy: Optional[tuple[float, float]] = None
    ignition_time: Optional[_dt.datetime] = None
    source: str = "unspecified"

    def __post_init__(self) -> None:
        self.burned_fraction_final = np.asarray(self.burned_fraction_final, dtype=float)
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        n_y, n_x = self.burned_fraction_final.shape
        if self.y.shape != (n_y,) or self.x.shape != (n_x,):
            raise ValueError(
                f"grid mismatch: burned_fraction_final {self.burned_fraction_final.shape}, "
                f"y {self.y.shape}, x {self.x.shape}"
            )
        if self.perimeter_times is not None or self.burned_fraction_series is not None:
            if self.perimeter_times is None or self.burned_fraction_series is None:
                raise ValueError("perimeter_times and burned_fraction_series must be given together")
            self.perimeter_times = [_to_datetime(t) for t in self.perimeter_times]
            self.burned_fraction_series = np.asarray(self.burned_fraction_series, dtype=float)
            if self.burned_fraction_series.shape != (len(self.perimeter_times), n_y, n_x):
                raise ValueError(
                    "burned_fraction_series must be (n_perimeter_times, n_y, n_x); got "
                    f"{self.burned_fraction_series.shape}"
                )
        self.ignition_time = _to_datetime(self.ignition_time)
        if self.ignition_xy is not None:
            self.ignition_xy = (float(self.ignition_xy[0]), float(self.ignition_xy[1]))

    @property
    def cell_area(self) -> float:
        return cell_area(self.x, self.y)

    def burned_mask_final(self, threshold: float = DEFAULT_FRACTION_THRESHOLD) -> np.ndarray:
        return self.burned_fraction_final >= threshold

    def regrid_to(self, x: np.ndarray, y: np.ndarray) -> "ObservedReference":
        """Nearest-neighbour resample onto another grid.

        Nearest-neighbour (not bilinear) keeps burned-fraction values and sharp
        perimeter edges intact — blurring a 0/1 footprint would invent partially
        burned cells along every boundary. Returns ``self`` unchanged when the
        grids already match.
        """
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if x.shape == self.x.shape and y.shape == self.y.shape and np.allclose(x, self.x) and np.allclose(y, self.y):
            return self
        ix = _nearest_index(self.x, x)
        iy = _nearest_index(self.y, y)
        sel = np.ix_(iy, ix)
        series = None
        if self.burned_fraction_series is not None:
            series = self.burned_fraction_series[:, iy][:, :, ix]
        return ObservedReference(
            burned_fraction_final=self.burned_fraction_final[sel],
            x=x,
            y=y,
            perimeter_times=self.perimeter_times,
            burned_fraction_series=series,
            ignition_xy=self.ignition_xy,
            ignition_time=self.ignition_time,
            source=self.source,
        )

    @classmethod
    def from_netcdf(
        cls,
        path: str | Path,
        *,
        fraction_var: str = "burned_fraction",
        x_var: str = "x",
        y_var: str = "y",
        time_var: str = "time",
        ignition_xy: Optional[tuple[float, float]] = None,
        ignition_time=None,
    ) -> "ObservedReference":
        """Read a rasterised observed reference from a NetCDF file.

        Matches the E1 loader output: ``burned_fraction(time, y, x)`` (the final
        footprint is the last time slice), or a static ``burned_fraction(y, x)``.
        ``netCDF4`` is imported lazily so the rest of the harness never depends on
        it.
        """
        import netCDF4  # lazy: only the file-driven path needs it

        path = Path(path)
        with netCDF4.Dataset(path) as ds:  # type: ignore[attr-defined]
            x = np.asarray(ds.variables[x_var][:], dtype=float)
            y = np.asarray(ds.variables[y_var][:], dtype=float)
            frac = np.asarray(ds.variables[fraction_var][:], dtype=float)
            times = None
            if time_var in ds.variables and frac.ndim == 3:
                tv = ds.variables[time_var]
                raw = np.asarray(tv[:])
                units = getattr(tv, "units", None)
                calendar = getattr(tv, "calendar", "standard")
                if units:
                    times = [
                        _to_datetime(netCDF4.num2date(  # type: ignore[attr-defined]
                            v, units, calendar, only_use_cftime_datetimes=False
                        ))
                        for v in raw
                    ]
        if frac.ndim == 2:
            final = frac
            series = None
        elif frac.ndim == 3:
            final = frac[-1]
            series = frac
        else:
            raise ValueError(f"{fraction_var} must be 2-D or 3-D, got {frac.ndim}-D")
        return cls(
            burned_fraction_final=final,
            x=x,
            y=y,
            perimeter_times=times if series is not None else None,
            burned_fraction_series=series,
            ignition_xy=ignition_xy,
            ignition_time=ignition_time,
            source=str(path),
        )


def _nearest_index(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Index into monotone ``src`` of the nearest value for each point in ``dst``."""
    src = np.asarray(src, float)
    order = np.argsort(src)
    src_sorted = src[order]
    pos = np.searchsorted(src_sorted, dst)
    pos = np.clip(pos, 1, len(src_sorted) - 1)
    left = src_sorted[pos - 1]
    right = src_sorted[pos]
    choose_left = (dst - left) <= (right - dst)
    idx_sorted = np.where(choose_left, pos - 1, pos)
    return order[idx_sorted]
