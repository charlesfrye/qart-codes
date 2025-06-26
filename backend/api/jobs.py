import base64
import time
from pathlib import Path
from typing import Optional

import modal

from .common import app
from .common import ASSETS_DIR, RESULTS_DIR, results_volume
from .datamodel import JobStatus, JobRequest


jobs = modal.Dict.from_name("qart-codes-jobs", create_if_missing=True)
# jobs.put("_test", {"status": JobStatus.COMPLETE})

Model = modal.Cls.from_name("qart-inference", "Model")
Aesthetics = modal.Cls.from_name("qart-eval", "Aesthetics")
Scannability = modal.Cls.from_name("qart-eval", "Scannability")

model = Model()
aesthetics = Aesthetics()
scannability = Scannability()


@app.local_entrypoint()  # for testing
async def main():
    from uuid import uuid4

    job_id = "_test"
    internal_id = uuid4()

    request = JobRequest(prompt="", image={"image_data": ""})
    print(f"Starting fake job with id {internal_id}...")

    await start(job_id, request)

    status = await check(job_id)
    assert status == JobStatus.PENDING

    await set_status(job_id, JobStatus.COMPLETE, payload={"_id": str(internal_id)})
    status = await check(job_id)

    if status == JobStatus.COMPLETE:
        payload = await read(job_id)
        assert payload["_id"] == str(internal_id)
        print(f"Response payload: {payload}")


async def start(job_id: str, request: JobRequest):
    try:
        await jobs.put.aio(job_id, {"status": JobStatus.PENDING, "handle": None})
        if job_id == "_test":
            return
        else:
            call = generate_and_save.spawn(
                job_id, request.prompt, request.image.image_data
            )
            # No-ops to cold start the evaluators
            aesthetics.wake.spawn(), scannability.wake.spawn()
            await jobs.put.aio(job_id, {"status": JobStatus.PENDING, "handle": call})

    except Exception as e:
        print(e)
        print(f"setting status of {job_id} to failed")
        await set_status(job_id, JobStatus.FAILED)
    pass


async def check(job_id: str) -> JobStatus:
    return await get_status(job_id)


async def cancel(job_id: str):
    await set_status(job_id, JobStatus.CANCELLED)


async def read(job_id: str) -> bytes:
    job = await jobs.get.aio(job_id)
    if job is None:
        raise KeyError(f"Job {job_id} not found")
    payload = job["payload"]
    await set_status(job_id, JobStatus.CONSUMED)
    return payload


@app.function(timeout=150, volumes={RESULTS_DIR: results_volume}, min_containers=1)
async def generate_and_save(job_id: str, prompt: str, image: str):
    """Generate a QR code from a prompt and push it into the jobs dict."""
    try:
        call = model.generate.spawn(prompt, image)
        await set_status(job_id, JobStatus.RUNNING)
        path = path_from_job_id(job_id)
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        await set_status(job_id, JobStatus.FAILED)
        jobs[job_id]["error"] = e
        return

    # await the result
    try:
        images_bytes = call.get(timeout=90)
    except TimeoutError as e:
        await set_status(job_id, JobStatus.FAILED)
        jobs[job_id]["error"] = e
        return

    # now call evaluators
    detecteds = [r async for r in scannability.check.map.aio(images_bytes)]
    ratings = [r async for r in aesthetics.score.map.aio(images_bytes)]

    # Construct the payload
    payload = [
        {
            "image": base64.b64encode(img).decode("utf-8"),
            "evaluation": {
                "detected": detecteds[ii][0] if detecteds[ii] else None,
                "aesthetic_rating": ratings[ii],
            },
        }
        for ii, img in enumerate(images_bytes)
    ]

    await set_status(job_id, JobStatus.COMPLETE, payload=payload)

    # write the result to a file
    await save_qr_codes(images_bytes, path)


async def save_qr_codes(images_bytes, basepath):
    import asyncio

    tasks = [
        save_qr_code(image_bytes, basepath / f"qr_{str(ii).zfill(2)}.png")
        for ii, image_bytes in enumerate(images_bytes)
    ]
    await asyncio.gather(*tasks)


async def save_qr_code(image_bytes, path):
    from aiofiles import open

    print(f"Saving: {path}")
    try:
        async with open(path, "wb") as f:
            await f.write(image_bytes)
    except Exception as e:
        print(f"Error saving {path}: {e}")


def path_from_job_id(job_id: str) -> Path:
    if job_id == "_test":
        return ASSETS_DIR
    else:
        parts = job_id.split("-")
        result_path = Path(RESULTS_DIR, *parts)
    return result_path


async def get_status(job_id: str) -> JobStatus:
    try:
        state = await jobs.get.aio(job_id)
        if state is None:
            return JobStatus.UNKNOWN
        return state["status"]
    except KeyError:
        print(job_id)
        return JobStatus.UNKNOWN


async def set_status(job_id: str, status: JobStatus, payload: Optional[bytes] = None):
    print(f"{job_id} setting status to {status}")
    state = await jobs.pop.aio(job_id)
    state["status"] = status
    state["payload"] = payload
    await jobs.put.aio(job_id, state)
