from pathlib import Path
from typing import Tuple
import modal
from common import app

DETECTOR_SET = Path(__file__).parent /  "evals" / "data" / "detector"

# we just share the same image across detectors
image = (modal.Image.debian_slim().
    apt_install("python3-opencv").
    pip_install("opencv-python", "qrcode", "pillow")
)

# shared function for generating qr code image
@app.function(image=image)
def generate_image(prompt: str, qr_url: str) -> bytes:
    from generator import Model
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
@app.function(image=image)
def detect_qr_opencv(image_bytes) -> Tuple[bool, str | None]:
    print("1")
    import cv2
    import os
    print("2")

    # lazy way to convert bytes to opencv image by writing to tmp file
    tmp_path = "tmp.png"
    with open(tmp_path, "wb") as f:
        f.write(image_bytes)
    image = cv2.imread(tmp_path)
    os.remove(tmp_path)
    print("3")

    if image is None:
        valid = False
        decoded_text = None
        return valid, decoded_text
        
    qr_detector = cv2.QRCodeDetector()
    decoded_text, points, _ = qr_detector.detectAndDecode(image)
    print("4")
    
    if points is None or decoded_text is None:
        valid = False
        decoded_text = None
        return valid, decoded_text

    valid = True
    return valid, decoded_text

# function for *other method* qr detection
@app.function(image=image)
def detect_qr_other(image_bytes) -> Tuple[bool, str | None]:
    # todo
    return False, None

# local entrypoint for sweeping through ./evals/data/detector 
@app.local_entrypoint()
def compare_detectors():
    # ensure detector set exists and is sorted
    if not DETECTOR_SET.exists():
        return

    user_instruction = "Detector set must be manually sorted into /valid and /invalid folders based on iphone qr reads"

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

    # opencv detector
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    for valid, _ in detect_qr_opencv.map(valid_img_bytes):
        if valid:
            print("True Positive")
            true_positives += 1
        else:
            print("False Negative")
            false_negatives += 1
    
    for valid, _ in detect_qr_opencv.map(invalid_img_bytes):
        if valid:
            print("False Positive")
            false_positives += 1
        else:
            print("True Negative")
            true_negatives += 1

    print("\nOpenCV:")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Negatives: {false_negatives}")