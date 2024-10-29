from pathlib import Path
import time

import requests


# BACKEND_URL = "https://charlesfrye--qart-api-dev.modal.run"
BACKEND_URL = "https://erik-dunteman--qart-api-dev.modal.run"
with open(Path("assets") / "qr-dataurl.txt") as f:
    test_qr_dataurl = f.read()

OUTPUT_PATH = Path(__file__).parent / "out"
OUTPUT_PATH.mkdir(exist_ok=True)


def test_end_to_end():
    health = requests.get(BACKEND_URL + "/health")
    health.raise_for_status()
    job_route = BACKEND_URL + "/job"
    start = time.monotonic_ns()
    result = requests.post(
        job_route,
        json={
            "prompt": "Constellation that could be interpreted as a bad omen",
            "image": {"image_data": test_qr_dataurl},
        },
    )
    job_posted = time.monotonic_ns()
    print("completed POST in", (job_posted - start) / 1e9, "seconds")
    result.raise_for_status()
    job_id = result.json()["job_id"]
    print(f"job_id: {job_id}")

    status = result.json()["status"]
    while status in ["PENDING", "RUNNING"]:
        time.sleep(1)
        result = requests.get(job_route, params={"job_id": job_id})
        result.raise_for_status()
        status = result.json()["status"]

    job_status_received = time.monotonic_ns()
    print("job status received in", (job_status_received - job_posted) / 1e9, "seconds")
    assert status == "COMPLETE", result

    result = requests.get(job_route + f"/{job_id}")
    result_received = time.monotonic_ns()
    print(
        "result received in", (result_received - job_status_received) / 1e9, "seconds"
    )
    result.raise_for_status()

    with open(OUTPUT_PATH / f"{job_id}-qr.png", "wb") as f:
        f.write(result.content)

    print("written to file at", f.name)
