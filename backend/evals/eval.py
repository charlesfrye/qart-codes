# parallel eval the outputs of testing

from pathlib import Path
import modal

image = modal.Image.debian_slim().apt_install("python3-opencv").pip_install("opencv-python", "qrcode", "pillow")

app = modal.App(name="qart-eval", image=image)

OUTPUT_PATH = Path(__file__).parent.parent / "tests" / "out"

with image.imports():
    import cv2
    import qrcode
    from typing import Union, Tuple
    from PIL import Image
    import numpy as np

@app.function()
def eval(image: bytes):
    tmp_path = "tmp.png"
    with open(tmp_path, "wb") as f:
        f.write(image)
    
    image = cv2.imread(tmp_path)
    if image is None:
        valid = False
        decoded_text = None
        return valid, decoded_text
        
    qr_detector = cv2.QRCodeDetector()
    decoded_text, points, _ = qr_detector.detectAndDecode(image)
    
    if points is None or decoded_text is None:
        valid = False
        decoded_text = None
        return valid, decoded_text
    
    valid = True
    return valid, decoded_text

@app.local_entrypoint()
def main():
    image_bytes = []
    image_paths = list(OUTPUT_PATH.glob("*.png"))
    for image_path in image_paths:
        image = image_path.read_bytes()
        image_bytes.append(image)

    # eval.remote(image_bytes[0])

    # run evals in parallel modal functions
    i = 0
    for valid, decoded_text in eval.map(image_bytes):
        image_path = str(image_paths[i]).split("/")[-1]
        print(f"Image: {image_path}\n\tValid: {valid}\n\tDecoded Text: {decoded_text}\n")
        i += 1