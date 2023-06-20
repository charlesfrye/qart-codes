"""Components, methods, and configuration used in multiple places."""
from pathlib import Path

import modal
from modal import Image, Mount, SharedVolume, Stub

ROOT_DIR = Path("/") / "root"
ASSETS_DIR = ROOT_DIR / "assets"
RESULTS_DIR = ROOT_DIR / "results"

toml_file_path = Path("pyproject.toml")
toml_file_mount = Mount.from_local_file(
    local_path=toml_file_path, remote_path=ROOT_DIR / toml_file_path
)

assets_mount = Mount.from_local_dir(local_path=Path("assets"), remote_path=ASSETS_DIR)

results_volume = SharedVolume().persist("qart-results-vol")

image = Image.debian_slim().pip_install("wonderwords", "Pillow")
stub = Stub("qart", image=image, mounts=[toml_file_mount, assets_mount])

if modal.is_local:
    with open(Path("assets") / "qr-dataurl.txt") as f:
        test_qr_dataurl = f.read()
else:
    with open(ASSETS_DIR / "qr-dataurl.txt") as f:
        test_qr_dataurl = f.read()
