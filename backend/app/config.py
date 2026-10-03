"""
RevuLens Backend Configuration.

Reads settings from environment variables with sensible defaults.
Supports .env files via python-dotenv.
"""

import os
from pathlib import Path
from typing import List

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass


class Settings:
    """Application settings with environment variable overrides."""

    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # Model & Artifact Paths
    MODEL_PATH: str = os.getenv(
        "SVM_MODEL_PATH",
        "backend/models/hybrid_distilbert_svm_pipeline.joblib"
    )
    DISTILBERT_CHECKPOINT: str = os.getenv(
        "DISTILBERT_MODEL_NAME",
        "distilbert-base-multilingual-cased"
    )

    # Explainability Defaults (per Step 6 latency benchmark decisions)
    DEFAULT_MAX_EVALS: int = int(os.getenv("SHAP_DEFAULT_MAX_EVALS", "60"))
    MAX_WORDS: int = int(os.getenv("SHAP_MAX_WORDS", "100"))
    CACHE_SIZE: int = int(os.getenv("SHAP_CACHE_SIZE", "256"))

    # CORS Origins (comma-separated string in env)
    _raw_cors: str = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000"
    )

    @property
    def CORS_ORIGINS(self) -> List[str]:
        """Parse comma-separated CORS origins into a list, stripping whitespace."""
        origins = [o.strip() for o in self._raw_cors.split(",") if o.strip()]
        # Allow any chrome-extension origin
        if "*" not in origins and "chrome-extension://*" not in origins:
            origins.append("chrome-extension://*")
        return origins


settings = Settings()
