import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env', override=False)


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is None:
        return None
    value = value.strip()
    return value or default


class Settings:
    DATABASE_URL = _get_env('DATABASE_URL')
    ALLOWED_ORIGINS = [
        origin.strip()
        for origin in (
            _get_env('ALLOWED_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173') or ''
        ).split(',')
        if origin.strip()
    ]

    @staticmethod
    def require_database_url() -> str:
        if not Settings.DATABASE_URL:
            raise RuntimeError(
                'DATABASE_URL is not configured. Copy .env.example to .env and set DATABASE_URL before starting the API.'
            )
        return Settings.DATABASE_URL


settings = Settings()
