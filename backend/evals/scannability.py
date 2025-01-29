from pathlib import Path
from typing import Optional

import modal

app = modal.App(name="test-qart-qreader")

here = Path(__file__).parent
ASSETS_DIR = here.parent / "assets"


@app.local_entrypoint()  # for testing
def main(
    image_path: str = str(ASSETS_DIR / "qart.png"),
    target_value: str = "https://tfs.ai/qart",
):
    print(f"Decoding QR code at {image_path}")
    detected, decoded_value = Scannability().check.remote(Path(image_path).read_bytes())
    if target_value:
        assert detected
        print("Detected QR code")
        assert (
            decoded_value == target_value
        ), f"Expected {target_value} but got {decoded_value}"
        print(f"Decoded QR code to {decoded_value}")


def download_model():
    from qrdet import QRDetector

    QRDetector()


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("python3-opencv", "libzbar0")
    .pip_install(
        "opencv-python==4.11.0.86",
        "pillow==11.1.0",
        "pyzbar==0.1.9",
        "qrdet==2.5",
        "qreader==3.14",
    )
    .run_function(download_model)
)


@app.cls(image=image, allow_concurrent_inputs=10)
class Scannability:
    def __init__(self):
        pass

    @modal.enter()
    def load(self):
        from qreader import QReader

        self.qreader = QReader()

    @modal.method()
    def wake(self):
        pass

    @modal.method()
    def check(self, image_bytes: bytes) -> tuple[bool, Optional[str]]:
        import cv2
        import numpy as np

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        opencv_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        decoded = self.qreader.detect_and_decode(image=image)
        if decoded is None or len(decoded) == 0 or decoded[0] is None:
            return False, None
        return True, decoded[0]
