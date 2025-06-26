"""Components, methods, and configuration used in multiple places."""

from pathlib import Path

import modal

ROOT_DIR = Path("/") / "root"
ASSETS_DIR = ROOT_DIR / "assets"
RESULTS_DIR = ROOT_DIR / "results"

here = Path(__file__).parent

toml_file_path = here.parent / "pyproject.toml"

results_volume = modal.Volume.from_name("qart-results-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi[standard]==0.115.13",
        "pydantic>=2,<3",
        "wonderwords==2.2.0",
        "Pillow==11.2.1",
        "aiofiles==24.1.0",
        "toml==0.10.2",
    )
    .add_local_file(
        local_path=toml_file_path, remote_path=str(ROOT_DIR / "pyproject.toml")
    )
    .add_local_dir(local_path=here.parent / "assets", remote_path=str(ASSETS_DIR))
)
app = modal.App("qart", image=image)

if modal.is_local:
    test_qr_dataurl_path = here.parent / "assets" / "qr-dataurl.txt"
else:
    test_qr_dataurl_path = ASSETS_DIR / "qr-dataurl.txt"

test_qr_dataurl = test_qr_dataurl_path.read_text()
