import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config, resolve_path  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def dev_sample_dir(cfg):
    return resolve_path(cfg, "paths.dev_sample")


@pytest.fixture(scope="session")
def dev_trips_path(dev_sample_dir):
    path = dev_sample_dir / "trips_clean.parquet"
    if not path.exists():
        pytest.skip("dev sample not built yet - run scripts/02_clean.py --dev-sample")
    return path


@pytest.fixture(scope="session")
def full_trips_path(cfg):
    path = resolve_path(cfg, "paths.interim") / "trips_clean.parquet"
    if not path.exists():
        pytest.skip("full clean not built yet - run scripts/02_clean.py")
    return path
