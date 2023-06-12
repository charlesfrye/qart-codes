import subprocess
import time

import requests


BACKEND_URL = "https://charlesfrye--qart-api-dev.modal.run"
test_qr_dataurl = """
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""".strip()


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
