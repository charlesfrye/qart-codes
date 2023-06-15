"""Components used in generative model and in API."""
from pathlib import Path

from modal import Image, Mount, SharedVolume, Stub

from .datamodel import JobStatus

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


def path_from_job_id(job_id: str) -> Path:
    if job_id == "_test":
        return ASSETS_DIR / "qart.png"
    else:
        parts = job_id.split("-")
        result_path = Path(RESULTS_DIR, *parts) / "qr.png"
    return result_path


def get_status(job_id: str) -> JobStatus:
    try:
        state = stub.app.jobs.get(job_id)
        return state["status"]
    except KeyError:
        print(job_id)
        return JobStatus.NOT_STARTED


def set_status(job_id: str, status: JobStatus):
    print(f"{job_id} setting status to {status}")
    state = stub.app.jobs.pop(job_id)
    state["status"] = status
    stub.app.jobs.put(job_id, state)
