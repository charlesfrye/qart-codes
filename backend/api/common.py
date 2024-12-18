"""Components, methods, and configuration used in multiple places."""

from pathlib import Path

import modal

ROOT_DIR = Path("/") / "root"
ASSETS_DIR = ROOT_DIR / "assets"
RESULTS_DIR = ROOT_DIR / "results"

here = Path(__file__).parent

toml_file_path = here.parent / "pyproject.toml"
toml_file_mount = modal.Mount.from_local_file(
    local_path=toml_file_path, remote_path=ROOT_DIR / "pyproject.toml"
)

assets_mount = modal.Mount.from_local_dir(
    local_path=here.parent / "assets", remote_path=ASSETS_DIR
)

results_volume = modal.Volume.from_name("qart-results-vol")

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "fastapi[standard]==0.115.5",
    "pydantic>=2,<3",
    "wonderwords",
    "Pillow",
    "aiofiles==24.1.0",
)
app = modal.App("qart", image=image, mounts=[toml_file_mount, assets_mount])

if modal.is_local:
    test_qr_dataurl_path = here.parent / "assets" / "qr-dataurl.txt"
else:
    test_qr_dataurl_path = ASSETS_DIR / "qr-dataurl.txt"

test_qr_dataurl = test_qr_dataurl_path.read_text()
