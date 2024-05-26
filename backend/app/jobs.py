from pathlib import Path

import modal
from modal import Dict

from .common import app
from .common import ASSETS_DIR, RESULTS_DIR, results_volume
from .datamodel import JobStatus, JobRequest
from .generator import Model

if modal.is_local:
    jobs = Dict.from_name(
        "qart-codes-test-jobs",
        {"_test": {"status": JobStatus.COMPLETE}},
        create_if_missing=True,
    )


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


def read(job_id: str):
    path = path_from_job_id(job_id)
    if not path.exists():
        raise FileNotFoundError
    return path


@app.function(
    timeout=150, network_file_systems={RESULTS_DIR: results_volume}, keep_warm=1
)
def generate_and_save(job_id: str, prompt: str, image: str):
    """Generate a QR code from a prompt and save it to a file."""
    try:
        path = path_from_job_id(job_id)
        dir_path = path.parent
        dir_path.mkdir(parents=True, exist_ok=True)
        call = Model().generate.spawn(prompt, image)
        set_status(job_id, JobStatus.RUNNING)
    except Exception as e:
        set_status(job_id, JobStatus.FAILED)
        jobs[job_id]["error"] = e
        return

    # await the result
    try:
        generator_response = call.get(timeout=90)
    except TimeoutError as e:
        set_status(job_id, JobStatus.FAILED)
        jobs[job_id]["error"] = e
        return

    # write the result to a file
    save_qr_code(generator_response, path)

    set_status(job_id, JobStatus.COMPLETE)


def save_qr_code(image, path):
    print(path)
    return image.save(path)


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
        return state["status"]
    except KeyError:
        print(job_id)
        return JobStatus.NOT_STARTED


def set_status(job_id: str, status: JobStatus):
    print(f"{job_id} setting status to {status}")
    state = jobs.pop(job_id)
    state["status"] = status
    jobs.put(job_id, state)
