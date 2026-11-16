# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Read access to the serving table (pipeline.serving) through DuckDB.

An area is a set of 0.1° cells, chosen by a bbox, a biome or a polygon registered
with `create_area`. A cell belongs to an area when its centre is inside. Every SQL
fragment built here comes from validated numbers or from a fixed list of biome
names, never from raw request text.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from app.history.analytics import Coverage

N_COLS = 3600  # 0.1° columns around the globe
CELLS_PER_DEGREE = 10
MAX_STORED_AREAS = 256
MAX_CACHED_SERIES = 32  # full-record daily series kept in memory, per area
LONG_REQUEST_DAYS = 730  # longer requests load (and cache) the area's whole record
COUNTRIES = {"BR": "Brazil"}  # cells labelled with any IBGE biome are Brazil


class AreaError(ValueError):
    """The area cannot be built (bad geometry, empty, unknown id)."""


class AreaTooLarge(AreaError):
    pass


@dataclass
class Area:
    kind: str  # bbox | biome | area_id
    cells: int
    predicate: str  # SQL condition on cell_id
    summary: dict = field(default_factory=dict)

    def as_summary(self) -> dict:
        return {"kind": self.kind, "cells": self.cells, **self.summary}


def cell_centres(rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half = 0.5 / CELLS_PER_DEGREE
    return rows / CELLS_PER_DEGREE - 90 + half, cols / CELLS_PER_DEGREE - 180 + half


def _rows_cols(w: float, s: float, e: float, n: float) -> tuple[int, int, int, int]:
    """Row/column range of the cells whose centre is inside the box."""
    half = 0.5 / CELLS_PER_DEGREE
    r0 = int(np.ceil((s + 90 - half) * CELLS_PER_DEGREE - 1e-9))
    r1 = int(np.floor((n + 90 - half) * CELLS_PER_DEGREE + 1e-9))
    c0 = int(np.ceil((w + 180 - half) * CELLS_PER_DEGREE - 1e-9))
    c1 = int(np.floor((e + 180 - half) * CELLS_PER_DEGREE + 1e-9))
    return r0, r1, c0, c1


def _points_in_ring(x: np.ndarray, y: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Even-odd rule: is each point (x, y) inside the closed ring (lon, lat)?"""
    inside = np.zeros(len(x), dtype=bool)
    xs, ys = ring[:, 0], ring[:, 1]
    for i in range(len(ring) - 1):
        x1, y1, x2, y2 = xs[i], ys[i], xs[i + 1], ys[i + 1]
        if y1 == y2:
            continue
        crosses = (y1 > y) != (y2 > y)
        x_at = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (x < x_at)
    return inside


def polygons_of(geometry: dict) -> list[list[np.ndarray]]:
    kind, coords = geometry.get("type"), geometry.get("coordinates")
    polys = [coords] if kind == "Polygon" else coords if kind == "MultiPolygon" else None
    if not polys:
        raise AreaError("geometry must be a GeoJSON Polygon or MultiPolygon")
    out = []
    try:
        for poly in polys:
            rings = [np.asarray(r, dtype=float)[:, :2] for r in poly]
            for r in rings:
                if len(r) < 4 or not np.isfinite(r).all():
                    raise AreaError("each ring needs at least 4 finite positions")
                if (np.abs(r[:, 0]) > 180).any() or (np.abs(r[:, 1]) > 90).any():
                    raise AreaError("coordinates must be lon/lat in degrees")
            out.append(rings)
    except (TypeError, ValueError, IndexError) as exc:
        if isinstance(exc, AreaError):
            raise
        raise AreaError("malformed GeoJSON coordinates") from exc
    return out


def cells_in_geometry(geometry: dict, max_cells: int) -> np.ndarray:
    polys = polygons_of(geometry)
    found = []
    for rings in polys:
        outer = rings[0]
        r0, r1, c0, c1 = _rows_cols(
            outer[:, 0].min(), outer[:, 1].min(), outer[:, 0].max(), outer[:, 1].max()
        )
        if r1 < r0 or c1 < c0:
            continue
        if (r1 - r0 + 1) * (c1 - c0 + 1) > 20 * max_cells:
            raise AreaTooLarge(f"polygon box is far above the limit of {max_cells} cells")
        rr, cc = np.meshgrid(np.arange(r0, r1 + 1), np.arange(c0, c1 + 1), indexing="ij")
        rr, cc = rr.ravel(), cc.ravel()
        lat, lon = cell_centres(rr, cc)
        inside = _points_in_ring(lon, lat, outer)
        for hole in rings[1:]:
            inside &= ~_points_in_ring(lon, lat, hole)
        found.append(rr[inside] * N_COLS + cc[inside])
    cells = np.unique(np.concatenate(found)) if found else np.array([], dtype=np.int64)
    if len(cells) > max_cells:
        raise AreaTooLarge(f"area has {len(cells)} cells; the limit is {max_cells}")
    if len(cells) == 0:
        raise AreaError("the polygon contains no 0.1° cell centre")
    return cells.astype(np.int64)


class HistoryStore:
    def __init__(self, data_dir: Path, memory_limit: str | None = None, threads: int | None = None):
        self.data_dir = Path(data_dir)
        meta_path = self.data_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} not found; run `python -m pipeline.serving`")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.limits = self.meta["limits"]
        self.con = duckdb.connect()
        if memory_limit:
            # Past the limit DuckDB spills to disk instead of failing or growing.
            spill = (Path(tempfile.gettempdir()) / "history_duckdb").as_posix()
            self.con.execute(f"SET memory_limit = '{memory_limit}'")
            self.con.execute(f"SET temp_directory = '{spill}'")
            self.con.execute("SET preserve_insertion_order = false")
        if threads:
            self.con.execute(f"SET threads = {int(threads)}")
        cell_day = (self.data_dir / "cell_day").as_posix()
        self.con.execute(
            "CREATE VIEW cell_day AS SELECT * FROM "
            f"read_parquet('{cell_day}/**/*.parquet', hive_partitioning = true)"
        )
        columns = {r[0] for r in self.con.execute("DESCRIBE cell_day").fetchall()}
        self._has_fine = "fine" in columns  # older tables have no VIIRS-only layer
        self._fine = "coalesce(sum(fine), 0)" if self._has_fine else "0.0"
        biome_file = (self.data_dir / "cell_biome.parquet").as_posix()
        self.con.execute(
            f"CREATE TABLE cell_biome AS SELECT cell_id, biome FROM read_parquet('{biome_file}')"
        )
        self.biome_cells = {b["name"]: b["cells"] for b in self.meta["biomes"]}
        self._areas: OrderedDict[str, Area] = OrderedDict()
        self._lock = threading.Lock()
        self.coverage = self._coverage()
        self._series: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._series_lock = threading.Lock()
        # One heavy scan at a time: concurrent requests (the frontend asks for one map per
        # year in parallel) would otherwise share DuckDB's memory limit and run it out.
        self._scan_lock = threading.Lock()
        cov = self.coverage
        ends = [cov.modis[1], cov.viirs[1]] + ([cov.estimate[1]] if cov.estimate else [])
        self.record = (min(cov.modis[0], cov.viirs[0]), max(ends))

    def _coverage(self) -> Coverage:
        c, h = self.meta["coverage"], self.meta["harmonization"]
        d = date.fromisoformat
        est = None
        if h.get("viirs_satellites"):
            est = (d(h["viirs_satellites"][0]["start"]), d(h["estimate_until"]))
        lay = h.get("layers") or {}
        layers = None
        if lay.get("status") == "available" and lay.get("start") and lay.get("end"):
            layers = (d(lay["start"]), d(lay["end"]))
        return Coverage(
            modis=(d(c["modis"]["start"]), d(c["modis"]["end"])),
            viirs=(d(c["viirs"]["start"]), d(c["viirs"]["end"])),
            estimate=est,
            layers=layers,
            switch_date=d(h["switch_date"]) if h.get("switch_date") else None,
            standard_until=min(d(c["modis"]["standard_until"]), d(c["viirs"]["standard_until"])),
        )

    # ---- areas -------------------------------------------------------------------
    def area_from_bbox(self, bbox: tuple[float, float, float, float]) -> Area:
        r0, r1, c0, c1 = _rows_cols(*bbox)
        cells = max(0, r1 - r0 + 1) * max(0, c1 - c0 + 1)
        if cells == 0:
            raise AreaError("the bbox contains no 0.1° cell centre")
        self._check_size(cells)
        pred = (
            f"(cell_id // {N_COLS}) BETWEEN {r0} AND {r1} "
            f"AND (cell_id % {N_COLS}) BETWEEN {c0} AND {c1}"
        )
        return Area("bbox", cells, pred, {"bbox": [float(v) for v in bbox]})

    def area_from_biome(self, biome: str) -> Area:
        if biome not in self.biome_cells:
            raise AreaError(f"unknown biome {biome}")
        cells = self.biome_cells[biome]
        self._check_size(cells)
        name = biome.replace("'", "")
        pred = f"cell_id IN (SELECT cell_id FROM cell_biome WHERE biome = '{name}')"
        return Area("biome", cells, pred, {"biome": biome})

    def area_from_country(self, code: str) -> Area:
        """A whole country. Predefined, so it is not held to `max_area_cells` (which
        bounds arbitrary boxes and polygons); its series is cached like a biome's."""
        if code not in COUNTRIES:
            raise AreaError(f"unknown country {code}")
        cells = sum(self.biome_cells.values())
        pred = "cell_id IN (SELECT cell_id FROM cell_biome WHERE biome IS NOT NULL)"
        return Area("country", cells, pred, {"country": code})

    def biome_cell_ids(self, biome: str) -> np.ndarray:
        """Ids of the 0.1° cells of a biome (used by the forecast outlook)."""
        if biome not in self.biome_cells:
            raise AreaError(f"unknown biome {biome}")
        rows = (
            self.con.cursor()
            .execute("SELECT cell_id FROM cell_biome WHERE biome = ?", [biome])
            .fetchnumpy()
        )
        return rows["cell_id"].astype(np.int64)

    def create_area(self, geometry: dict) -> tuple[Area, dict]:
        cells = cells_in_geometry(geometry, self.limits["max_area_cells"])
        area_id = hashlib.sha1(cells.tobytes()).hexdigest()[:16]
        table = f"area_{area_id}"
        with self._lock:
            if area_id not in self._areas:
                frame = pd.DataFrame({"cell_id": cells})  # noqa: F841 (read by DuckDB)
                self.con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM frame")
                self._areas[area_id] = Area(
                    "area_id",
                    len(cells),
                    f"cell_id IN (SELECT cell_id FROM {table})",
                    {"area_id": area_id},
                )
                while len(self._areas) > MAX_STORED_AREAS:
                    old, _ = self._areas.popitem(last=False)
                    self.con.execute(f"DROP TABLE IF EXISTS area_{old}")
            self._areas.move_to_end(area_id)
        lat, lon = cell_centres(cells // N_COLS, cells % N_COLS)
        half = 0.5 / CELLS_PER_DEGREE
        bbox = [
            round(float(lon.min() - half), 2),
            round(float(lat.min() - half), 2),
            round(float(lon.max() + half), 2),
            round(float(lat.max() + half), 2),
        ]
        shares = self.biome_shares(self._areas[area_id])
        return self._areas[area_id], {"bbox": bbox, "biomes": shares}

    def area_from_id(self, area_id: str) -> Area:
        with self._lock:
            area = self._areas.get(area_id)
        if area is None:
            raise KeyError(area_id)
        return area

    def _check_size(self, cells: int) -> None:
        if cells > self.limits["max_area_cells"]:
            raise AreaTooLarge(
                f"area has {cells} cells; the limit is {self.limits['max_area_cells']}"
            )

    def biome_shares(self, area: Area) -> dict[str, float]:
        rows = (
            self.con.cursor()
            .execute(f"""
            WITH a AS (
                SELECT cell_id FROM cell_biome WHERE {area.predicate}
            )
            SELECT coalesce(b.biome, 'outside'), count(*)
            FROM a JOIN cell_biome b USING (cell_id) GROUP BY 1
        """)
            .fetchall()
        )
        known = sum(n for _, n in rows)
        shares = {b: n / area.cells for b, n in rows}
        if area.cells > known:  # cells outside the labelled grid
            shares["outside"] = shares.get("outside", 0) + (area.cells - known) / area.cells
        return {b: round(v, 4) for b, v in sorted(shares.items())}

    def estimate_band(self, area: Area) -> tuple[float, float]:
        """90% band of the VIIRS estimate for this area: the monthly per-biome bands
        weighted by the area's share of each biome (an approximation)."""
        return self._band(area, self.meta["harmonization"].get("interval_90_monthly_by_biome"))

    def detected_band(self, area: Area) -> tuple[float, float]:
        """90% band of `detected` (estimate + VIIRS-only layer) for this area, from its
        monthly errors against the observed union on the reference years."""
        check = (self.meta["harmonization"].get("layers") or {}).get("check") or {}
        return self._band(area, check.get("interval_90_monthly_by_biome"))

    def _band(self, area: Area, bands: dict | None) -> tuple[float, float]:
        bands = bands or {}
        shares = {b: s for b, s in self.biome_shares(area).items() if b in bands}
        total = sum(shares.values())
        if not total:
            return (0.0, 0.0)
        lo = sum(bands[b][0] * s for b, s in shares.items()) / total
        hi = sum(bands[b][1] * s for b, s in shares.items()) / total
        return (lo, hi)

    def cell_biome_frame(self) -> pd.DataFrame:
        """Every labelled 0.1° cell: `cell_id` and `biome` (None outside Brazil)."""
        return self.con.cursor().execute("SELECT cell_id, biome FROM cell_biome").fetchdf()

    # ---- data --------------------------------------------------------------------
    def daily(self, area: Area, start: date, end: date) -> pd.DataFrame:
        """Area sums per day: modis, viirs (observed), est (VIIRS estimate), fine
        (VIIRS-only layer); every day of [start, end] is present (0 when nothing happened).

        Biomes, countries and long requests read the area's whole record once and keep
        it in memory (LRU of MAX_CACHED_SERIES areas); later requests slice it. The data
        is read-only for the life of the process, so the cache never goes stale."""
        key = area.predicate
        with self._series_lock:
            full = self._series.get(key)
            if full is not None:
                self._series.move_to_end(key)
        if full is None and (
            area.kind in ("biome", "country") or (end - start).days > LONG_REQUEST_DAYS
        ):
            full = self._query_daily(area, *self.record)
            with self._series_lock:
                self._series[key] = full
                while len(self._series) > MAX_CACHED_SERIES:
                    self._series.popitem(last=False)
        if full is None:
            return self._query_daily(area, start, end)
        idx = pd.date_range(start, end, freq="D")
        return full.reindex(idx, fill_value=0.0)

    def warm(self) -> None:
        """Load the series of every biome and country (called in the background at start)."""
        areas = [self.area_from_biome(b) for b in self.biome_cells] + [
            self.area_from_country(c) for c in COUNTRIES
        ]
        for area in areas:
            self.daily(area, *self.record)

    def _query_daily(self, area: Area, start: date, end: date) -> pd.DataFrame:
        """Area sums per day, straight from the serving table (see `daily`)."""
        with self._scan_lock:
            rows = self._daily_rows(area, start, end)
        idx = pd.date_range(start, end, freq="D")
        rows["date"] = pd.to_datetime(rows["date"])
        frame = rows.set_index("date").reindex(idx, fill_value=0)
        return frame.astype({"modis": float, "viirs": float, "est": float, "fine": float})

    def _daily_rows(self, area: Area, start: date, end: date) -> pd.DataFrame:
        return (
            self.con.cursor()
            .execute(f"""
            SELECT date, sum(modis) AS modis, sum(viirs) AS viirs, coalesce(sum(est), 0) AS est,
                   {self._fine} AS fine
            FROM cell_day
            WHERE year BETWEEN {start.year} AND {end.year}
              AND date BETWEEN DATE '{start.isoformat()}' AND DATE '{end.isoformat()}'
              AND {area.predicate}
            GROUP BY date
        """)
            .fetchdf()
        )

    def per_cell(self, area: Area, start: date, end: date) -> pd.DataFrame:
        """Per cell with any fire in [start, end]: modis and viirs observed counts and the
        harmonized value (MODIS days before the switch, estimate days after)."""
        with self._scan_lock:
            return self._per_cell(area, start, end)

    def _per_cell(self, area: Area, start: date, end: date) -> pd.DataFrame:
        cov = self.coverage
        m_end = cov.harmonized_modis_end().isoformat()
        e0, e1 = cov.estimate if cov.estimate else (date.max, date.min)
        return (
            self.con.cursor()
            .execute(f"""
            SELECT cell_id,
                   sum(modis) AS modis,
                   sum(viirs) AS viirs,
                   sum(CASE WHEN date <= DATE '{m_end}' THEN modis
                            WHEN date BETWEEN DATE '{e0.isoformat()}' AND DATE '{e1.isoformat()}'
                            THEN coalesce(est, 0) ELSE 0 END) AS harmonized,
                   {"coalesce(sum(fine), 0)" if self._has_fine else "0.0"} AS viirs_only
            FROM cell_day
            WHERE year BETWEEN {start.year} AND {end.year}
              AND date BETWEEN DATE '{start.isoformat()}' AND DATE '{end.isoformat()}'
              AND {area.predicate}
            GROUP BY cell_id
            HAVING sum(modis) > 0 OR sum(viirs) > 0 OR sum(coalesce(est, 0)) > 0
            ORDER BY cell_id
        """)
            .fetchdf()
        )
