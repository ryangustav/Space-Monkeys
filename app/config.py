# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Brazil plus a margin: west, south, east, north.
DEFAULT_BBOX = (-74.5, -34.5, -28.5, 6.0)

DEFAULT_FIRMS_SOURCES = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT"]


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables or `.env`.

    List values are comma-separated in the environment, e.g.
    `FIRMS_SOURCES=VIIRS_SNPP_NRT,MODIS_NRT` or `AOI_BBOX=-74.5,-34.5,-28.5,6`.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # NASA FIRMS area API (https://firms.modaps.eosdis.nasa.gov/api/area/)
    firms_map_key: str = ""
    firms_sources: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(DEFAULT_FIRMS_SOURCES)
    )
    firms_day_range: int = Field(default=1, ge=1, le=5)

    # NOAA GOES-R ABI L2 Fire/Hot Spot Characterization, Full Disk (public AWS bucket)
    goes_bucket: str = "noaa-goes19"
    goes_product: str = "ABI-L2-FDCF"
    goes_min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Area monitored by the real-time worker
    aoi_bbox: Annotated[tuple[float, float, float, float], NoDecode] = DEFAULT_BBOX

    # Real-time worker
    enable_worker: bool = True
    poll_interval_seconds: int = Field(default=600, ge=60)
    live_retention_hours: float = Field(default=48.0, gt=0)

    # Multi-sensor fusion
    fusion_radius_km: float = Field(default=2.0, gt=0)
    fusion_window_hours: float = Field(default=12.0, gt=0)

    # Burning activity calendar: folder written by `python -m pipeline.serving`
    history_data_dir: Path = Path("data/processed/brazil/serving")
    # Cache the daily series of every biome and of Brazil in the background at start-up.
    history_warm_up: bool = False
    # DuckDB would otherwise size itself to the host RAM, not the 1 GB container.
    history_memory_limit: str = "400MB"
    history_threads: int = Field(default=2, ge=1)

    # Fire forecast (U-Net, see notebooks/fire_forecast_base.py)
    enable_forecast: bool = True
    forecast_model_dir: str = "models/fire_forecast_brazil"
    forecast_refresh_hours: float = Field(default=6.0, gt=0)
    forecast_anchor_hours: float = Field(default=12.0, gt=0)
    # FIRMS NRT source fed to the model (S-NPP, the training satellite, ended 2026-11-02)
    forecast_source: str = "VIIRS_NOAA20_NRT"

    # HTTP
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["*"]  # public read-only API, no cookies or auth
    )
    sse_heartbeat_seconds: float = Field(default=15.0, gt=0)

    @field_validator("firms_sources", "cors_origins", mode="before")
    @classmethod
    def _split_list(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("aoi_bbox", mode="before")
    @classmethod
    def _split_bbox(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(float(item) for item in value.split(","))
        return value

    @model_validator(mode="after")
    def _check_bbox(self) -> "Settings":
        w, s, e, n = self.aoi_bbox
        if not (w < e and s < n):
            raise ValueError("AOI_BBOX must be west,south,east,north with west<east, south<north")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


def redact(text: str) -> str:
    """Remove secrets (the FIRMS map key) from text shown in logs or API responses."""
    key = get_settings().firms_map_key
    return text.replace(key, "***") if key else text
