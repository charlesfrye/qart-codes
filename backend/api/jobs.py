from pathlib import Path
from typing import Optional

import modal

from .common import app
from .common import ASSETS_DIR, RESULTS_DIR, results_volume
from .datamodel import JobStatus, JobRequest

jobs = modal.Dict.from_name(
    "qart-codes-jobs",
    {"_test": {"status": JobStatus.COMPLETE}},
    create_if_missing=True,
)

Model = modal.Cls.lookup("qart-inference", "Model")


def start(job_id: str, request: JobRequest):
    try:
        if job_id == "_test":
            return
        else:
            jobs.put(job_id, {"status": JobStatus.PENDING, "handle": None})
            call = generate_and_save.spawn(
                job_id, request.prompt, request.image.image_data
            )
            jobs.put(job_id, {"status": JobStatus.PENDING, "handle": call})

    except Exception as e:
        print(e)
        print(f"setting status of {job_id} to failed")
        set_status(job_id, JobStatus.FAILED)
    pass


def check(job_id: str) -> JobStatus:
    return get_status(job_id)


def cancel(job_id: str):
    set_status(job_id, JobStatus.CANCELLED)


def read(job_id: str) -> bytes:
    job = jobs.get(job_id)
    if job is None:
        raise KeyError(f"Job {job_id} not found")
    payload = job["payload"]
    set_status(job_id, JobStatus.CONSUMED)
    return payload


@app.function(
    timeout=150, network_file_systems={RESULTS_DIR: results_volume}, keep_warm=1
)
def generate_and_save(job_id: str, prompt: str, image: str):
    """Generate a QR code from a prompt and push it into the jobs dict."""
    try:
        call = Model().generate.spawn(prompt, image)
        set_status(job_id, JobStatus.RUNNING)
        path = path_from_job_id(job_id)
        dir_path = path.parent
        dir_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        set_status(job_id, JobStatus.FAILED)
        jobs[job_id]["error"] = e
        return

    # await the result
    try:
        image_bytes = call.get(timeout=90)
    except TimeoutError as e:
        set_status(job_id, JobStatus.FAILED)
        jobs[job_id]["error"] = e
        return

    set_status(job_id, JobStatus.COMPLETE, payload=image_bytes)

    # write the result to a file
    save_qr_code(image_bytes, path)


def save_qr_code(image_bytes, path):
    print(path)
    with open(path, "wb") as f:
        f.write(image_bytes)


def path_from_job_id(job_id: str) -> Path:
    if job_id == "_test":
        return ASSETS_DIR / "qart.png"
    else:
        parts = job_id.split("-")
        result_path = Path(RESULTS_DIR, *parts) / "qr.png"
    return result_path


def get_status(job_id: str) -> JobStatus:
    try:
        state = jobs.get(job_id)
        if state is None:
            return JobStatus.UNKNOWN
        return state["status"]
    except KeyError:
        print(job_id)
        return JobStatus.UNKNOWN


def set_status(job_id: str, status: JobStatus, payload: Optional[bytes] = None):
    print(f"{job_id} setting status to {status}")
    state = jobs.pop(job_id)
    state["status"] = status
    state["payload"] = payload
    jobs.put(job_id, state)
