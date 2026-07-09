from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    DB_PATH: str = "data/scheduler.db"

    # Cluster
    TOTAL_GPUS: int = 8

    # Fair-share accounting.
    # Usage decays with a half-life so that recent consumption dominates and old
    # consumption fades. Without decay, a user who ran heavy jobs months ago would
    # be penalised forever.
    HALF_LIFE_HOURS: float = 168.0  # 7 days

    # Priority = W_FAIR * fairshare_factor + W_AGE * age_factor
    #
    # fairshare_factor is bounded in (0, 1]; age_factor grows without bound, one
    # unit per AGE_UNIT_HOURS queued. The unbounded term is what guarantees no
    # job starves: a long enough wait outweighs any fair-share deficit.
    W_FAIR: float = 0.7
    W_AGE: float = 0.3
    AGE_UNIT_HOURS: float = 24.0

    # Scheduler loop
    TICK_SECONDS: float = 1.0

    # Guardrails
    MAX_GPUS_PER_JOB: int = 8
    MAX_EST_SECONDS: int = 60 * 60 * 24


settings = Settings()
