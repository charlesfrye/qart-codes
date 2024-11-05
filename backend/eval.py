from pathlib import Path
from typing import Tuple
import modal
from common import app

from generator import Model

DETECTOR_SET = Path(__file__).parent /  "evals" / "data" / "detector"

# we just share the same image across detectors
image = (modal.Image.debian_slim().
    apt_install("python3-opencv", "libzbar0").
    pip_install("opencv-python", "qrcode", "pillow", "pyzbar", "qreader")
)

# shared function for generating qr code image
@app.function(
    image=image, 
    # concurrency_limit=1
)
def generate_image(prompt: str, qr_url: str) -> bytes:

    import base64
    import io
    import qrcode

    # produces pil image from url
    image = qrcode.make(qr_url) 
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    # put into base64 (format expected by modal)
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # generate image
    model = Model()
    image_bytes = model.generate.remote(prompt, image_base64)

    return image_bytes

# local entrypoint for generating ./evals/data/detector set
@app.local_entrypoint()
def generate_detector_set():
    from wonderwords import RandomWord # requires pip install wonderwords
    import random

    urls = [
        "https://www.google.com",
        "https://www.youtube.com",
        "https://www.facebook.com",
        "https://www.twitter.com",
        "https://www.reddit.com",
        "https://www.instagram.com",
        "https://www.pinterest.com",
        "https://www.linkedin.com",
        "https://www.tiktok.com",
        "https://www.snapchat.com",
        "https://www.twitch.tv",
        "https://www.discord.com",
        "https://www.spotify.com",
        "https://www.soundcloud.com",
        "https://www.bandcamp.com",
        "https://www.vimeo.com",
        "https://www.flickr.com",
        "https://www.behance.net",
        "https://www.dribbble.com",
        "https://www.deviantart.com",
        "https://www.500px.com",
    ]

    args = [] # for starmap, tuple of (prompt, qr_url)
    for i in range(100):
        url = urls[random.randint(0, len(urls) - 1)]
        r_gen = RandomWord()
        adjective1 = r_gen.word(include_parts_of_speech=["adjective"])
        adjective2 = r_gen.word(include_parts_of_speech=["adjective"])
        adjective3 = r_gen.word(include_parts_of_speech=["adjective"])
        noun1 = r_gen.word(include_parts_of_speech=["noun"])
        noun2 = r_gen.word(include_parts_of_speech=["noun"])
        prompt = f"A {adjective1} {noun1} in a {adjective2} {adjective3} {noun2}"
        args.append((prompt, url))
    
    i = 0
    for image_bytes in generate_image.starmap(args):
        prompt = args[i][0].replace(" ", "_")
        i += 1
        image_path = DETECTOR_SET / f"{prompt}.png"
        with open(image_path, "wb") as f:
            f.write(image_bytes)



# function for openCV qr detection
@app.function(image=image, allow_concurrent_inputs=100)
def detect_qr_opencv(image_bytes) -> Tuple[bool, str | None]:
    import cv2
    import os
    from uuid import uuid4

    # lazy way to convert bytes to opencv image by writing to tmp file
    # we allow concurrent inputs, so use uuid for the file to avoid collisions
    uuid = str(uuid4())
    tmp_path = f"tmp_{uuid}.png"
    with open(tmp_path, "wb") as f:
        f.write(image_bytes)
    image = cv2.imread(tmp_path)
    os.remove(tmp_path)

    if image is None:
        valid = False
        decoded_text = None
        return valid, decoded_text
        
    qr_detector = cv2.QRCodeDetector()
    decoded_text, points, _ = qr_detector.detectAndDecode(image)
    
    if points is None or decoded_text is None or decoded_text == "":
        valid = False
        decoded_text = None
        return valid, decoded_text

    valid = True
    return valid, decoded_text

# function for pyzbar qr detection
@app.function(image=image, allow_concurrent_inputs=100)
def detect_qr_pyzbar(image_bytes) -> Tuple[bool, str | None]:
    import cv2
    import os
    from uuid import uuid4
    from pyzbar import pyzbar

    # lazy way to convert bytes to opencv image by writing to tmp file
    # we allow concurrent inputs, so use uuid for the file to avoid collisions
    uuid = str(uuid4())
    tmp_path = f"tmp_{uuid}.png"
    with open(tmp_path, "wb") as f:
        f.write(image_bytes)
    image = cv2.imread(tmp_path)
    os.remove(tmp_path)

    if image is None:
        valid = False
        decoded_text = None
        return valid, decoded_text  

    barcodes = pyzbar.decode(image)
    if len(barcodes) == 0:
        valid = False
        decoded_text = None
        return valid, decoded_text

    if len(barcodes) > 1:
        print("Found multiple QR codes, defaulting to first")

    barcode = barcodes[0]
    barcode_data = barcode.data.decode("utf-8")
    if barcode_data == "" or barcode_data is None:
        valid = False
        decoded_text = None
        return valid, decoded_text
    
    valid = True
    return valid, barcode_data

# QReader uses YOLO model under the hood to detect QR codes
# So we do this one as a modal class rather than a function,
# to reuse the model weights between calls
@app.cls(
    image=image,
    gpu="t4",
)
class QReader:
    def __init__(self, threshold = 0.5, model_size = "s"):
        self.threshold = threshold
        self.model_size = model_size
    
    @modal.build() # assuming it downloads and caches the model on load
    @modal.enter()
    def load(self):
        from qreader import QReader as QR
        self.qreader = QR(
            min_confidence = self.threshold, 
            model_size = self.model_size
        )

    @modal.method()
    def detect_and_decode(self, image_bytes):
        import cv2
        import os
        from uuid import uuid4

        # lazy way to convert bytes to opencv image by writing to tmp file
        # we allow concurrent inputs, so use uuid for the file to avoid collisions
        uuid = str(uuid4())
        tmp_path = f"tmp_{uuid}.png"
        with open(tmp_path, "wb") as f:
            f.write(image_bytes)
        image = cv2.imread(tmp_path)
        os.remove(tmp_path)

        if image is None:
            valid = False
            decoded_text = None  
            return valid, decoded_text
        
        decoded_text = self.qreader.detect_and_decode(image=image)
        if len(decoded_text) == 0:
            valid = False
            decoded_text = None
            return valid, decoded_text

        decoded_text = decoded_text[0]
        return True, decoded_text

# local entrypoint for sweeping through ./evals/data/detector
@app.local_entrypoint()
def compare_detectors():
    # ensure detector set exists and is sorted
    if not DETECTOR_SET.exists():
        return
    
    
    user_instruction = "Detector set must be manually sorted into /valid and /invalid folders based on iphone qr reads." # noqa: E501

    unsorted = list(DETECTOR_SET.glob("*.png"))
    if len(unsorted) != 0:
        print(user_instruction)

    valid_dir = DETECTOR_SET / "valid"
    invalid_dir = DETECTOR_SET / "invalid"

    # ensure valid and invalid dirs exist
    if not valid_dir.exists() or not invalid_dir.exists():
        print(user_instruction)
        return

    # ensure valid and invalid dirs are not empty
    valid_imgs = list(valid_dir.glob("*.png"))
    invalid_imgs = list(invalid_dir.glob("*.png"))

    if len(valid_imgs) == 0 or len(invalid_imgs) == 0:
        print(user_instruction)
        return
    
    valid_img_bytes = []
    invalid_img_bytes = []

    for valid_img in valid_imgs:
        with open(valid_img, "rb") as f:
            valid_img_bytes.append(f.read())
    
    for invalid_img in invalid_imgs:
        with open(invalid_img, "rb") as f:
            invalid_img_bytes.append(f.read())

    # load our QReader detector, which is class-based so needs to pre-init
    qreader = QReader()

    # map of detector functions. Must have the same function signatures!
    detectors = {
        "OpenCV": detect_qr_opencv, 
        "Pyzbar": detect_qr_pyzbar, 
        "QReader": qreader.detect_and_decode
    }

    for detector_name, detector in detectors.items():
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        for valid, _ in detector.map(valid_img_bytes):
            if valid:
                true_positives += 1
            else:
                false_negatives += 1
        
        for valid, _ in detector.map(invalid_img_bytes):
            if valid:
                false_positives += 1
            else:
                true_negatives += 1

        print(f"\nDetector:        {detector_name}")
        print(f"True Positives:  {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"True Negatives:  {true_negatives}")
        print(f"False Negatives: {false_negatives}")

@app.local_entrypoint()
def optimize_qreader_threshold():
    # knowing that qreader has an adjustable threshold, 
    # we can use an optimization library to find the best threshold
    # this is likely overkill, but it's a fun exercise!
    
    import numpy as np
    from scipy.optimize import minimize

    valid_dir = DETECTOR_SET / "valid"
    invalid_dir = DETECTOR_SET / "invalid"
    valid_imgs = list(valid_dir.glob("*.png"))
    invalid_imgs = list(invalid_dir.glob("*.png"))
    valid_img_bytes = []
    invalid_img_bytes = []
    for valid_img in valid_imgs:
        with open(valid_img, "rb") as f:
            valid_img_bytes.append(f.read())
    for invalid_img in invalid_imgs:
        with open(invalid_img, "rb") as f:
            invalid_img_bytes.append(f.read())
    
    def objective(threshold):
        threshold = threshold[0] # Scipy provides a tuple to objective func
        print(f"\nTesting threshold: {threshold:.3f}")
        qreader = QReader(threshold = threshold)
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        for valid, _ in qreader.detect_and_decode.map(valid_img_bytes):
            if valid:
                true_positives += 1
            else:
                false_negatives += 1
        
        for valid, _ in qreader.detect_and_decode.map(invalid_img_bytes):
            if valid:
                false_positives += 1
            else:
                true_negatives += 1

        print(f"True Positives:  {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"True Negatives:  {true_negatives}")
        print(f"False Negatives: {false_negatives}")
        true_cases = true_positives + true_negatives
        false_cases = false_positives + false_negatives
        score = true_cases / (true_cases + false_cases)
        print(f"Score:           {score:.3f}")
        return -score # Scipy can only minimize, so we negate the score

    # Phase 1: Coarse grid search
    print("Beginning coarse grid search...")
    thresholds = np.linspace(0.5, 0.99, 8)  # 8 points between thresholds 0.5 and 0.99
    best_score = -2 # initial score, theoretical min is -1
    best_threshold = None
    for threshold in thresholds:
        score = -objective([threshold])
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    # Phase 2: Fine-tune with L-BFGS-B
    print("Beginning fine grain optimization...")
    result = minimize(
        objective,
        best_threshold,
        bounds=[(max(0, best_threshold-0.2), min(1, best_threshold+0.2))],  # Narrow bounds
        method='L-BFGS-B',
        options={'maxiter': 8}  # Just a few iterations for fine-tuning
    )

    optimal_threshold = result.x[0]
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"Best score: {-result.fun:.3f}")


@app.local_entrypoint()
def test_ensemble():
    # if we keep the OR of opencv and pyzbar, does that eliminate false negatives and get closer to iphone?
    
    valid_dir = DETECTOR_SET / "valid"
    invalid_dir = DETECTOR_SET / "invalid"
    valid_imgs = list(valid_dir.glob("*.png"))
    invalid_imgs = list(invalid_dir.glob("*.png"))
    valid_img_bytes = []
    invalid_img_bytes = []
    for valid_img in valid_imgs:
        with open(valid_img, "rb") as f:
            valid_img_bytes.append(f.read())
    for invalid_img in invalid_imgs:
        with open(invalid_img, "rb") as f:
            invalid_img_bytes.append(f.read())
        

    opencv_detected = []
    pyzbar_detected = []
    for valid, _ in detect_qr_opencv.map(valid_img_bytes + invalid_img_bytes):
        opencv_detected.append(valid)
    for valid, _ in detect_qr_pyzbar.map(valid_img_bytes + invalid_img_bytes):
        pyzbar_detected.append(valid)

    print("OpenCV:")
    print([1 if valid else 0 for valid in opencv_detected])
    print("Pyzbar:")
    print([1 if valid else 0 for valid in pyzbar_detected])

    ensemble_detected = [x or y for x, y in zip(opencv_detected, pyzbar_detected)]
    print("Ensemble:")
    print([1 if valid else 0 for valid in ensemble_detected])

    def score(name, detections):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i, valid in enumerate(detections):
            if i < len(valid_imgs):
                # then we're in the valid bucket
                if valid:
                    tp += 1
                else:
                    fn += 1
            else:
                # then we're in the invalid bucket
                if valid:
                    fp += 1
                else:
                    tn += 1

        print(f"For detector {name}:")
        print(f"True Positives:  {tp}")
        print(f"False Positives: {fp}")
        print(f"True Negatives:  {tn}")
        print(f"False Negatives: {fn}")
        print(f"Score:           {(tp + tn) / (tp + tn + fp + fn):.3f}")
    
    score("OpenCV", opencv_detected)
    score("Pyzbar", pyzbar_detected)
    score("Ensemble", ensemble_detected)

# function for ensemble qr detection
@app.function(image=image, allow_concurrent_inputs=100)
def detect_qr_ensemble(image_bytes) -> Tuple[bool, str | None]:
    import cv2
    import os
    from uuid import uuid4
    from pyzbar import pyzbar

    # lazy way to convert bytes to opencv image by writing to tmp file
    # we allow concurrent inputs, so use uuid for the file to avoid collisions
    uuid = str(uuid4())
    tmp_path = f"tmp_{uuid}.png"
    with open(tmp_path, "wb") as f:
        f.write(image_bytes)
    image = cv2.imread(tmp_path)
    os.remove(tmp_path)

    if image is None:
        valid = False
        decoded_text = None
        return valid, decoded_text  

    # OpenCV
    qr_detector = cv2.QRCodeDetector()
    opencv_decoded_text, opencv_points, _ = qr_detector.detectAndDecode(image)
    opencv_detected = opencv_points is not None and opencv_decoded_text is not None and opencv_decoded_text != ""

    # Pyzbar
    pyz_barcodes = pyzbar.decode(image)
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


@app.local_entrypoint()
def eval_scannability():
    # The official eval script for scannability
    # We use the ensemble of opencv and pyzbar as our detector

    prompts = [
        "A cat in a hat",
        # "A pirate ship birthday party",
        # "A shiba inu drinking an americano",
        # "Solarpunk architecture",
    ]

    n = 10

    for prompt in prompts:
        print(f"Prompt: {prompt}")
        # use google as the qr url for now
        for image_bytes in generate_image.starmap([(prompt, f"https://www.google.com")] * n):
            valid, decoded_text = detect_qr_ensemble.remote(image_bytes)
            if valid:
                print(f"Valid QR code: {decoded_text}")
            else:
                print(f"Invalid QR code: {decoded_text}")