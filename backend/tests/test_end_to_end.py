import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
import requests

load_dotenv()


BACKEND_URL = (
    os.environ.get("DEV_BACKEND_URL") or "https://charlesfrye--qart-api-dev.modal.run"
)
here = Path(__file__).parent

test_qr_dataurl = (here.parent / "assets" / "qr-dataurl.txt").read_text()

output_dir = here / "out"
output_dir.mkdir(exist_ok=True)


def test_end_to_end():
    health_response = requests.get(BACKEND_URL + "/health")
    health_response.raise_for_status()

    job_route = BACKEND_URL + "/job"
    start = time.monotonic_ns()
    result = requests.post(
        job_route,
        json={
            "prompt": "a Shiba Inu drinking an Americano and eating pancakes",
            "image": {"image_data": test_qr_dataurl},
        },
    )
    job_posted = time.monotonic_ns()
    print("completed POST in", (job_posted - start) / 1e9, "seconds")
    result.raise_for_status()
    job_id = result.json()["job_id"]
    print(f"job_id: {job_id}")

    status = result.json()["status"]
    while status in ["PENDING", "RUNNING", "UNKNOWN"]:
        time.sleep(0.1)
        result = requests.get(job_route, params={"job_id": job_id})
        result.raise_for_status()
        status = result.json()["status"]

    job_status_complete = time.monotonic_ns()
    print("job marked complete in", (job_status_complete - job_posted) / 1e9, "seconds")
    assert status == "COMPLETE", result

    result = requests.get(job_route + f"/{job_id}")
    result_received = time.monotonic_ns()
    print(
        "result received in", (result_received - job_status_complete) / 1e9, "seconds"
    )
    result.raise_for_status()

    output_path = output_dir / f"{job_id}-response.jsonl"
    for ii, res in enumerate(result.json()):
        print(f"{ii}: {res['evaluation']}")
    output_path.write_text(
        "\n".join(json.dumps(result) for result in result.json()) + "\n"
    )

    print("written to file at", output_path)


if __name__ == "__main__":
    test_end_to_end()
