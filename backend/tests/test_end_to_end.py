from pathlib import Path
import subprocess
import time

import requests


BACKEND_URL = "https://charlesfrye--qart-api-dev.modal.run"
with open(Path("assets") / "qr-dataurl.txt") as f:
    test_qr_dataurl = f.read()


def test_end_to_end():
    job_route = BACKEND_URL + "/job"
    result = requests.post(
        job_route,
        json={
            "prompt": "a Shiba Inu drinking an Americano and eating pancakes",
            "image": {"image_data": test_qr_dataurl},
        },
    )
    result.raise_for_status()
    job_id = result.json()["job_id"]
    print(f"job_id: {job_id}")

    status = result.json()["status"]
    while status in ["PENDING", "RUNNING"]:
        time.sleep(1)
        result = requests.get(job_route, params={"job_id": job_id})
        result.raise_for_status()
        status = result.json()["status"]

    assert status == "COMPLETE", result

    result = requests.get(job_route + f"/{job_id}")
    result.raise_for_status()

    subprocess.check_output(
        [
            "modal",
            "volume",
            "get",
            "--force",
            "qart-results-vol",
            job_id.replace("-", "/") + "/qr.png",
            f"tests/{job_id}-qr.png",
        ]
    )
