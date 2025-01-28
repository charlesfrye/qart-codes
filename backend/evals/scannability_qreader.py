import modal

app = modal.App(name="qart-qreader")


def download_model():
    from qrdet import QRDetector

    QRDetector()  # Specify model size here to download non-default ('s' - small)


image = (
    modal.Image.debian_slim()
    .apt_install("python3-opencv", "libzbar0")
    .pip_install("opencv-python", "pillow", "pyzbar", "qrdet", "qreader")
    .run_function(download_model)
)


@app.cls(image=image, gpu="any", allow_concurrent_inputs=10)
class TunedQReader:
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
    def detect_qr_qreader(self, image_bytes):
        import cv2
        import numpy as np

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        opencv_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        decoded = self.qreader.detect_and_decode(image=image)
        if decoded is None or len(decoded) == 0 or decoded[0] is None:
            return False, None
        return True, decoded[0]