# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""NOAA GOES-East ABI L2 Fire/Hot Spot Characterization (FDC) reader.

The Full Disk product (ABI-L2-FDCF) covers South America every 10 minutes at ~2 km
(nadir). Files are public on AWS through the NOAA Open Data Dissemination program:
https://noaa-goes19.s3.amazonaws.com/ABI-L2-FDCF/{year}/{day-of-year}/{hour}/...

Pixel positions come from the GOES-R fixed-grid projection described in the GOES-R
Product Definition and User's Guide (PUG), Volume 5, section 4.2.8.
"""

import asyncio
import io
import logging
import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta

import httpx
import numpy as np
import pandas as pd
import xarray as xr

from app.realtime import detections

log = logging.getLogger(__name__)

S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

# Fire mask categories (PUG Vol. 5, Fire/Hot Spot Characterization).
# 10-15 instantaneous, 30-35 the same classes temporally filtered.
MASK_CONFIDENCE = {
    10: 0.9,  # good quality fire pixel
    11: 0.9,  # saturated fire pixel
    12: 0.6,  # cloud contaminated fire pixel
    13: 0.7,  # high probability fire pixel
    14: 0.5,  # medium probability fire pixel
    15: 0.3,  # low probability fire pixel
}
MASK_CONFIDENCE.update({code + 20: conf for code, conf in MASK_CONFIDENCE.items()})
# The fire mask is scanned in bands of rows so a full disk never sits in memory at once.
MASK_BAND_ROWS = 512

_START_RE = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})")


def scan_start(key: str) -> datetime:
    """Scan start time from a GOES file name (`..._sYYYYJJJHHMMSSt_...`)."""
    match = _START_RE.search(key)
    if not match:
        raise ValueError(f"Not a GOES file name: {key}")
    year, doy, hh, mm, ss = (int(g) for g in match.groups())
    return datetime(year, 1, 1, hh, mm, ss, tzinfo=UTC) + timedelta(days=doy - 1)


def satellite_id(key: str) -> str:
    match = re.search(r"_(G\d{2})_", key)
    return match.group(1) if match else "G??"


async def list_keys(client: httpx.AsyncClient, bucket: str, prefix: str) -> list[str]:
    url = f"https://{bucket}.s3.amazonaws.com/"
    keys: list[str] = []
    params: dict[str, str] = {"list-type": "2", "prefix": prefix}
    while True:
        response = await client.get(url, params=params, timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        keys += [el.text for el in root.findall("s3:Contents/s3:Key", S3_NS) if el.text]
        token = root.find("s3:NextContinuationToken", S3_NS)
        if token is None or not token.text:
            return keys
        params["continuation-token"] = token.text


async def latest_key(
    client: httpx.AsyncClient, bucket: str, product: str, now: datetime | None = None
) -> str | None:
    """Newest file of the product, looking back up to 2 hours."""
    now = now or datetime.now(UTC)
    for hours_back in range(3):
        t = now - timedelta(hours=hours_back)
        prefix = f"{product}/{t:%Y}/{t.timetuple().tm_yday:03d}/{t:%H}/"
        keys = [k for k in await list_keys(client, bucket, prefix) if k.endswith(".nc")]
        if keys:
            return max(keys, key=scan_start)
    return None


def fixed_grid_to_latlon(x: np.ndarray, y: np.ndarray, proj: dict) -> tuple[np.ndarray, np.ndarray]:
    """GOES-R fixed grid scan angles (radians) to geodetic lat/lon (degrees).

    Points off the Earth's disk come back as NaN.
    """
    r_eq = float(proj["semi_major_axis"])
    r_pol = float(proj["semi_minor_axis"])
    h = float(proj["perspective_point_height"]) + r_eq
    lon0 = np.deg2rad(float(proj["longitude_of_projection_origin"]))

    sin_x, cos_x = np.sin(x), np.cos(x)
    sin_y, cos_y = np.sin(y), np.cos(y)
    ratio = (r_eq / r_pol) ** 2

    a = sin_x**2 + cos_x**2 * (cos_y**2 + ratio * sin_y**2)
    b = -2.0 * h * cos_x * cos_y
    c = h**2 - r_eq**2
    disc = b**2 - 4.0 * a * c
    with np.errstate(invalid="ignore"):
        r_s = (-b - np.sqrt(disc)) / (2.0 * a)
    r_s = np.where(disc < 0, np.nan, r_s)

    s_x = r_s * cos_x * cos_y
    s_y = -r_s * sin_x
    s_z = r_s * cos_x * sin_y

    lat = np.arctan(ratio * s_z / np.sqrt((h - s_x) ** 2 + s_y**2))
    lon = lon0 - np.arctan(s_y / (h - s_x))
    return np.rad2deg(lat), np.rad2deg(lon)


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def pixel_footprint_km(
    x: np.ndarray, y: np.ndarray, dx: float, dy: float, proj: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Ground size of each pixel (east-west, north-south), from its corners' positions.

    Pixels grow away from the sub-satellite point: about 2 km at nadir, roughly 3-5 km
    over most of Brazil.
    """
    lat_w, lon_w = fixed_grid_to_latlon(x - dx / 2, y, proj)
    lat_e, lon_e = fixed_grid_to_latlon(x + dx / 2, y, proj)
    lat_n, lon_n = fixed_grid_to_latlon(x, y + dy / 2, proj)
    lat_s, lon_s = fixed_grid_to_latlon(x, y - dy / 2, proj)
    return haversine_km(lat_w, lon_w, lat_e, lon_e), haversine_km(lat_n, lon_n, lat_s, lon_s)


def parse_fdc(
    data: bytes,
    key: str,
    bbox: tuple[float, float, float, float],
    min_confidence: float = 0.5,
) -> pd.DataFrame:
    """Extract fire pixels from one FDC NetCDF file into the common detections table.

    Memory: a full-disk grid is 5424 × 5424. Decoded by xarray, each 2D variable becomes
    a float64 array of about 235 MB, enough to push a 1 GB server out of memory. So the
    mask is read raw (int16, ~59 MB, it has no scale), fire pixels outside the area are
    dropped from their coordinates alone, and Power is read only in the window that
    holds the remaining pixels."""
    fire_codes = np.array(list(MASK_CONFIDENCE))
    rows, cols, found = [], [], []
    with xr.open_dataset(io.BytesIO(data), engine="h5netcdf", mask_and_scale=False) as raw:
        band = raw["Mask"]
        for r in range(0, band.shape[0], MASK_BAND_ROWS):
            mask = band[r : r + MASK_BAND_ROWS].values
            by, bx = np.nonzero(np.isin(mask, fire_codes))
            rows.append(by + r)
            cols.append(bx)
            found.append(mask[by, bx])
    iy, ix = np.concatenate(rows), np.concatenate(cols)
    codes = np.concatenate(found).astype(int)
    if iy.size == 0:
        return detections.empty()

    with xr.open_dataset(io.BytesIO(data), engine="h5netcdf") as ds:
        x_all = ds["x"].values.astype("float64")
        y_all = ds["y"].values.astype("float64")
        proj = dict(ds["goes_imager_projection"].attrs)
        lat, lon = fixed_grid_to_latlon(x_all[ix], y_all[iy], proj)
        w, s, e, n = bbox
        confidence = np.array([MASK_CONFIDENCE[c] for c in codes])
        keep = (
            np.isfinite(lat)
            & (lat >= s)
            & (lat <= n)
            & (lon >= w)
            & (lon <= e)
            & (confidence >= min_confidence)
        )
        if not keep.any():
            return detections.empty()
        iy, ix, codes, lat, lon = iy[keep], ix[keep], codes[keep], lat[keep], lon[keep]
        r0, c0 = int(iy.min()), int(ix.min())
        window = ds["Power"][r0 : int(iy.max()) + 1, c0 : int(ix.max()) + 1].values
        power = window[iy - r0, ix - c0].astype("float64")

    x, y = x_all[ix], y_all[iy]
    dx = float(abs(x_all[1] - x_all[0]))
    dy = float(abs(y_all[1] - y_all[0]))
    scan_km, track_km = pixel_footprint_km(x, y, dx, dy, proj)

    confidence = np.array([MASK_CONFIDENCE[c] for c in codes])
    df = pd.DataFrame(
        {
            "source": f"GOES:{satellite_id(key)}",
            "sensor": "ABI",
            "satellite": satellite_id(key),
            "time": pd.Timestamp(scan_start(key)),
            "lat": lat,
            "lon": lon,
            "scan_km": scan_km,
            "track_km": track_km,
            "frp_mw": power,
            "confidence": confidence,
            "daynight": "",
        }
    )
    df = df.dropna(subset=["lat", "lon"])
    df = df[df["confidence"] >= min_confidence]
    df = detections.in_bbox(df, bbox)
    return detections.finalize(df)


async def fetch_latest(
    client: httpx.AsyncClient,
    bucket: str,
    product: str,
    bbox: tuple[float, float, float, float],
    min_confidence: float,
    skip_key: str | None = None,
) -> tuple[pd.DataFrame, str | None]:
    """Download and parse the newest FDC file. Returns (detections, key).

    When the newest key equals `skip_key` nothing is downloaded.
    """
    key = await latest_key(client, bucket, product)
    if key is None or key == skip_key:
        return detections.empty(), key
    response = await client.get(f"https://{bucket}.s3.amazonaws.com/{key}", timeout=120)
    response.raise_for_status()
    df = await asyncio.to_thread(parse_fdc, response.content, key, bbox, min_confidence)
    return df, key
