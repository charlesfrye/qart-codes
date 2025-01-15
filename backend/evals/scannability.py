import modal
from typing import Tuple

app = modal.App(name="qart-scannability")

# we just share the same image across detectors
image = (modal.Image.debian_slim().
    apt_install("python3-opencv", "libzbar0").
    pip_install("opencv-python", "qrcode", "pillow", "pyzbar", "qreader")
)

@app.function(image=image, allow_concurrent_inputs=100)
def detect_qr_ensemble(image_bytes: bytes) -> Tuple[bool, str | None]:
    """Detect QR codes using OpenCV and pyzbar.
    Args:
        image_bytes (bytes): QR Image in bytes.
    Returns:
        Tuple[bool, str | None]: (valid, decoded_text)
    
    """
    import io
    import cv2
    import numpy as np
    from PIL import Image
    from pyzbar import pyzbar

    # OpenCV
    image_array = np.frombuffer(image_bytes, dtype=np.uint8) 
    opencv_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    qr_detector = cv2.QRCodeDetector()
    try:
        opencv_decoded_text, opencv_points, _ = qr_detector.detectAndDecode(opencv_image)
    except:
        opencv_decoded_text = None
        opencv_points = None

    opencv_detected = opencv_points is not None and opencv_decoded_text is not None and opencv_decoded_text != ""

    # Pyzbar
    pil_image = Image.open(io.BytesIO(image_bytes))
    pyz_barcodes = pyzbar.decode(pil_image)
    pyzbar_detected = len(pyz_barcodes) > 0
    
    # If neither detect, return false
    if not pyzbar_detected and not opencv_detected:
        valid = False
        decoded_text = None
        return valid, decoded_text

    # If just pyzbar detects, return pyzbar result
    if pyzbar_detected and not opencv_detected:
        decoded_text = pyz_barcodes[0].data.decode("utf-8")
        return True, decoded_text
    
    # If just opencv detects, return opencv result
    if opencv_detected and not pyzbar_detected:
        return True, opencv_decoded_text

    # If both detect, ensure they're the same
    if opencv_detected and pyzbar_detected:
        if pyz_barcodes[0].data.decode("utf-8") == opencv_decoded_text:
            return True, pyz_barcodes[0].data.decode("utf-8")
        else:
            print(f"Pyzbar and OpenCV decode to different texts, {pyz_barcodes[0].data.decode('utf-8')} and {opencv_decoded_text} returning false")
            return False, None
