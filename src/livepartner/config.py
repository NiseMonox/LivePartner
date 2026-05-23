from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model_flash: str
    model_pro: str

    @classmethod
    def from_env(cls) -> "Settings":
        key = os.environ.get("LP_LLM_API_KEY", "").strip()
        if not key:
            raise RuntimeError("LP_LLM_API_KEY not set — check .env")
        base = os.environ.get("LP_LLM_BASE_URL", "").strip()
        if not base:
            raise RuntimeError("LP_LLM_BASE_URL not set — check .env")
        flash = os.environ.get("LP_MODEL_FLASH", "").strip()
        pro = os.environ.get("LP_MODEL_PRO", "").strip()
        if not flash or not pro:
            raise RuntimeError("LP_MODEL_FLASH and LP_MODEL_PRO must both be set")
        return cls(api_key=key, base_url=base, model_flash=flash, model_pro=pro)


settings = Settings.from_env()
