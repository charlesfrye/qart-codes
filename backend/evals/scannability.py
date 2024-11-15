import modal
from pathlib import Path
from typing import Tuple

from .common import app, generate_image

SCANNABILITY_SET = Path(__file__).parent /  "data" / "scannability"

# we just share the same image across detectors
image = (modal.Image.debian_slim().
    apt_install("python3-opencv", "libzbar0").
    pip_install("opencv-python", "qrcode", "pillow", "pyzbar", "qreader")
)

################################################################################
# Scannability eval
# Below is the best scannability eval function we landed on.
# See lower in this file for experiments run to choose this.
################################################################################

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


################################################################################
# Experiments while developing the scannability eval
# These are not used in the final eval, but are useful for showing the process
################################################################################

# local entrypoint for generating ./data/scannability set
# from quart root:
# modal run backend.evals.scannability::generate_detector_set
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
        image_path = SCANNABILITY_SET / f"{prompt}.png"
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
        self.model_size = model_size # note model caching at build will only cache the small model
    
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

# local entrypoint for testing ./data/scannability with different detectors
# from quart root:
# modal run backend.evals.scannability::compare_detectors
@app.local_entrypoint()
def compare_detectors():
    # ensure detector set exists and is sorted
    if not SCANNABILITY_SET.exists():
        return
    
    
    user_instruction = "Detector set must be manually sorted into /valid and /invalid folders based on iphone qr reads." # noqa: E501

    unsorted = list(SCANNABILITY_SET.glob("*.png"))
    if len(unsorted) != 0:
        print(user_instruction)

    valid_dir = SCANNABILITY_SET / "valid"
    invalid_dir = SCANNABILITY_SET / "invalid"

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
    qreader_s = QReader() # defaults to small model
    qreader_n = QReader(model_size = "n") # nano model

    # map of detector functions. Must have the same function signatures!
    detectors = {
        "OpenCV": detect_qr_opencv, 
        "Pyzbar": detect_qr_pyzbar, 
        "QReader (Small)": qreader_s.detect_and_decode,
        "QReader (Nano)": qreader_n.detect_and_decode
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

# local entrypoint for optimizing qreader threshold
# from quart root:
# modal run backend.evals.scannability::optimize_qreader_threshold
@app.local_entrypoint()
def optimize_qreader_threshold(size = "s"):
    # knowing that qreader has an adjustable threshold, 
    # we can use an optimization library to find the best threshold
    # this is likely overkill, but it's a fun exercise!
    
    import numpy as np
    from scipy.optimize import minimize

    valid_dir = SCANNABILITY_SET / "valid"
    invalid_dir = SCANNABILITY_SET / "invalid"
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
        qreader = QReader(threshold = threshold, model_size = size)
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

# local entrypoint for testing an ensemble of opencv and pyzbar
# from quart root:
# modal run backend.evals.scannability::test_ensemble
@app.local_entrypoint()
def test_ensemble():
    # if we keep the OR of opencv and pyzbar, does that eliminate false negatives and get closer to iphone?
    
    valid_dir = SCANNABILITY_SET / "valid"
    invalid_dir = SCANNABILITY_SET / "invalid"
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

# local entrypoints for testing various ensemble strategies
# from quart root:
# modal run backend.evals.scannability::generate_detection_table
DETECTION_TABLE_PATH = Path(__file__).parent / "data" / "scannability" / "detection_table.json"
QREADER_THRESHOLDS = [0.86, 0.88, 0.9, 0.92, 0.94, 0.96] 
@app.local_entrypoint()
def generate_detection_table():
    import json
    # Ensemble seems to improve! We're getting few false positives, but a lot of false negatives.
    # The idea was raised to use yolo, only in the case of a negative, which means there's potential to do branching logic in our ensembling.
    # Let's generate a table of all of our detections, so we can programmatically test which ensemble strategy is best.
    valid_dir = SCANNABILITY_SET / "valid"
    invalid_dir = SCANNABILITY_SET / "invalid"
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

    table = {
        "iphone": [1] * len(valid_imgs) + [0] * len(invalid_imgs),
    }
    
    # openCV
    opencv_detected = []
    for valid, _ in detect_qr_opencv.map(valid_img_bytes + invalid_img_bytes):
        opencv_detected.append(1 if valid else 0)
    table["opencv"] = opencv_detected

    # pyzbar
    pyzbar_detected = []
    for valid, _ in detect_qr_pyzbar.map(valid_img_bytes + invalid_img_bytes):
        pyzbar_detected.append(1 if valid else 0)
    table["pyzbar"] = pyzbar_detected

    # QReader at various thresholds
    for threshold in QREADER_THRESHOLDS:
        qreader_detected = []
        qreader = QReader(threshold = threshold)
        for valid, _ in qreader.detect_and_decode.map(valid_img_bytes + invalid_img_bytes):
            qreader_detected.append(1 if valid else 0)
        table[f"qreader@{threshold}"] = qreader_detected
    
    # save to file
    with open(DETECTION_TABLE_PATH, "w") as f:
        json.dump(table, f)

# from quart root:
# modal run backend.evals.scannability::score_detection_table
@app.local_entrypoint()
def score_detection_table():
    import json
    # load the detection table
    with open(DETECTION_TABLE_PATH, "r") as f:
        table = json.load(f)

    # policies at each detected value
    # we assume we only use one threshold of QReader
    def policy_opencv(opencv, pyzbar, qreader):
        return opencv
    def policy_pyzbar(opencv, pyzbar, qreader):
        return pyzbar
    def policy_qreader(opencv, pyzbar, qreader):
        return qreader

    # ensemble
    def policy_opencv_pyzbar_ensemble(opencv, pyzbar, qreader):
        if opencv == 1 or pyzbar == 1:
            return 1
        return 0

    # tacking yolo only onto negatives
    def policy_opencv_w_qreader_backup(opencv, pyzbar, qreader):
        if opencv == 1:
            return 1
        return qreader
    def policy_pyzbar_w_qreader_backup(opencv, pyzbar, qreader):
        if pyzbar == 1:
            return 1
        return qreader
    def policy_ensemble_w_qreader_backup(opencv, pyzbar, qreader):
        if opencv == 1 or pyzbar == 1:
            return 1
        return qreader

    # F-beta score
    def calculate_fbeta_score(tp, fp, fn, beta=0.5):
        """Calculate F-beta score with given beta value.
        
        beta < 1 weights precision more heavily than recall
        beta > 1 weights recall more heavily than precision
        beta = 1 weights precision and recall equally (standard F1 score)
        """
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if precision == 0 and recall == 0:
            return 0
        
        beta_squared = beta * beta
        fbeta = ((1 + beta_squared) * precision * recall) / \
                (beta_squared * precision + recall) if (beta_squared * precision + recall) > 0 else 0
        
        return fbeta, precision, recall
    
    # calculate stats for each policy
    policy_scores = {}
    for policy_name, policy in [
        ("opencv", policy_opencv),
        ("pyzbar", policy_pyzbar),
        ("qreader", policy_qreader),
        ("opencv_pyzbar_ensemble", policy_opencv_pyzbar_ensemble),
        ("opencv_w_qreader_backup", policy_opencv_w_qreader_backup),
        ("pyzbar_w_qreader_backup", policy_pyzbar_w_qreader_backup),
        ("ensemble_w_qreader_backup", policy_ensemble_w_qreader_backup),
        ]:
        for threshold in QREADER_THRESHOLDS:
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for i in range(len(table["iphone"])):
                iphone = table["iphone"][i]
                policy_score = policy(
                    table["opencv"][i],
                    table["pyzbar"][i],
                    table[f"qreader@{threshold}"][i]
                )
                if iphone == 1:
                    if policy_score == 1:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if policy_score == 1:
                        fp += 1
                    else:
                        tn += 1
            fbeta_score, precision, recall = calculate_fbeta_score(tp, fp, fn, beta=0.5)
            policy_scores[policy_name + f"@{threshold}"] = {
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f0.5": fbeta_score,
                "accuracy": (tp + tn) / (tp + tn + fp + fn),
            }

    # rank policies by score
    ranked_policies = sorted(policy_scores.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    print("Top 5 policies by accuracy: TP + TN / (TP + TN + FP + FN)")
    for policy_name, policy_score in ranked_policies[:5]:
        print(f"{policy_name}: {policy_score}")
    
    # rank policies by precision
    ranked_policies = sorted(policy_scores.items(), key=lambda x: x[1]["precision"], reverse=True)
    print("\nTop 5 policies by precision: TP / (TP + FP)")
    for policy_name, policy_score in ranked_policies[:5]:
        print(f"{policy_name}: {policy_score}")

    # rank policies by recall
    ranked_policies = sorted(policy_scores.items(), key=lambda x: x[1]["recall"], reverse=True)
    print("\nTop 5 policies by recall: TP / (TP + FN)")
    for policy_name, policy_score in ranked_policies[:5]:
        print(f"{policy_name}: {policy_score}")

    # rank policies by F0.5
    ranked_policies = sorted(policy_scores.items(), key=lambda x: x[1]["f0.5"], reverse=True)
    print("\nTop 5 policies by F0.5:")
    for policy_name, policy_score in ranked_policies[:5]:
        print(f"{policy_name}: {policy_score}")