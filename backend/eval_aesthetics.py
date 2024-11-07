from pathlib import Path
from typing import Tuple
import modal
from common import app
from uuid import uuid4
import os

from generator import Model

AESTHETICS_SET = Path(__file__).parent /  "evals" / "data" / "aesthetics"
SCANNABILITY_SET = Path(__file__).parent /  "evals" / "data" / "scannability"

# we just share the same image across detectors
image = (modal.Image.debian_slim()
    .apt_install("git", "wget")
    .pip_install(
        "qrcode", 
        "Pillow", 
        "transformers", 
        "torch", 
        "pytorch-lightning",
        "git+https://github.com/openai/CLIP.git"
    )
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

@app.local_entrypoint()
def copy_scannability_set():
    import shutil
    # First, copy in the sorted scannability set, no use in generating the same images twice
    valid_dir = SCANNABILITY_SET / "valid"
    invalid_dir = SCANNABILITY_SET / "invalid"
    dest = AESTHETICS_SET
    valid_imgs = list(valid_dir.glob("*.png"))
    invalid_imgs = list(invalid_dir.glob("*.png"))
    for img in valid_imgs:
        # copy into aesthetics set
        shutil.copy(img, dest)
    for img in invalid_imgs:
        # copy into aesthetics set
        shutil.copy(img, dest)

# need multiple URLs so the idea of aesthetics doesn't accidentally confuse a specific QR as aesthetic
URLS = [
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

# local entrypoint for generating ./evals/data/aesthetics set
@app.local_entrypoint()
def generate_aesthetics_set_random():
    from wonderwords import RandomWord # requires pip install wonderwords
    import random

    args = [] # for starmap, tuple of (prompt, qr_url)
    for i in range(10):
        url = URLS[random.randint(0, len(URLS) - 1)]
        r_gen = RandomWord()
        adjective1 = r_gen.word(include_parts_of_speech=["adjective"])
        noun1 = r_gen.word(include_parts_of_speech=["noun"])
        
        prompt = f"A {adjective1} {noun1}, 3D render in blender, trending on artstation, uses depth to trick the eye, colorful, high resolution, 8k, cinematic, concept art, illustration, painting, modern"
        args.append((prompt, url))

    i = 0
    for image_bytes in generate_image.starmap(args):
        prompt = args[i][0].replace(" ", "_")
        i += 1
        image_path = AESTHETICS_SET / f"{prompt}.png"
        with open(image_path, "wb") as f:
            f.write(image_bytes)

@app.local_entrypoint()
def generate_aesthetics_set_manual():
    import random
    import os

    # to target more aesthetic images, no more random nouns and adjectives
    prompts = [
        "A hillside italian city, vivid colors and shadows, with a church tower and bustling fishing port. High resolution, realistic, artistic.",
        "An isometric solarpunk cityscape, vivid and green with futuristic buildings and a flying car in the foreground. High resolution, realistic, artistic.",
        "Pirates on a desert island, with a ship in the foreground. Clean, white sails show a jolly roger. Pixel art style.",
        "Ancient temple ruins with stone archways and columns at different depths, sunbeams streaming through gaps, moss-covered stonework. Photorealistic, golden hour lighting.",
        "Japanese zen garden with wooden bridges at different levels, stone lanterns, and a pagoda in the background. Soft morning mist, architectural photography style.",
        "Art deco hotel lobby with grand staircases, crystal chandeliers, and marble columns creating natural frames. Vintage photography style, warm lighting.",
        "Geometric rock formations in Utah desert, multiple arches at different distances, small figure for scale. Sunset lighting, landscape photography.",
        "Misty redwood forest with fallen logs and wooden walkways at various depths, shaft of sunlight breaking through. Moody atmospheric photography.",
        "Terraced rice fields in Bali at sunrise, with traditional buildings and palm trees creating natural layers. Aerial photography perspective.",
        "Abandoned factory interior with catwalks at different levels, broken windows, and ivy growing inside. Urban exploration photography style.",
        "Modern glass skyscraper atrium with floating staircases, indoor plants, and people on different levels. Architecture visualization style.",
        "Underground metro station with multiple platforms, escalators, and geometric ceiling patterns. Long exposure photography style.",
        "M.C. Escher-inspired impossible architecture with staircases going in different directions, rendered in a clean modern style.",
        "Geometric abstract composition with floating cubes and spheres at different depths, inspired by Kandinsky. Bold colors, clean lines.",
        "Steampunk clockwork mechanism with gears and pistons at multiple layers, brass and copper tones. Technical illustration style.",
        "Chess tournament from above, with multiple boards at different depths, players leaning over tables. Documentary photography style.",
        "Busy night market with food stalls, lanterns, and crowds creating natural depth layers. Cyberpunk color palette.",
        "Library interior with spiral staircases, books floating between floors, and reading nooks at different levels. Magical realism style.",
        "Retro-futuristic computer room with mainframes and data centers at different depths, inspired by 1970s sci-fi.",
        "Holographic display room with floating screens at various distances, subtle blue glow. Clean minimal sci-fi style.",
        "Quantum computer laboratory with suspended magnetic rings and laser arrays at different depths. Scientific visualization style.",
    ]

    refined_prompts = [
        "Ancient Roman bathhouse interior, steam rising between marble columns, sunlight streaming through high windows onto different pool levels. Architectural photography.",
        "Venetian canal at sunset, gondolas passing under arched bridges, laundry lines crossing between buildings, warm light reflecting off water. Traditional photography.",
        "Traditional Arabic souq, domed skylights casting patterns through hanging lamps, merchants and carpets creating layers in narrow passages. Documentary style.",
        "Mountain monastery carved into cliff face, connected by covered wooden walkways, prayer flags stretching across different levels, morning fog in valleys. Landscape photography.",
        "Victorian greenhouse interior, wrought iron staircases between plant levels, sunlight filtering through fogged glass panels, tropical plants creating natural screens. Historical documentation style."
    ]

    refined_prompts_2 = [
        "Vintage record store browsing scene, wooden crates and album displays at various heights, customers flipping through vinyl, warm pendant lights casting shadows. Documentary style.",
        "Artist's loft workspace, easels and canvases arranged in rows, paint splattered dropsheets creating texture, afternoon light through industrial windows. Interior photography.",
        "Cozy library reading nook, leather armchairs and standing lamps creating intimate spaces, books stacked on circular tables, bay window lighting. Lifestyle photography.",
        "Chess tournament in progress, rows of players hunched over boards, tournament clocks and scoresheets creating patterns, overhead lighting casting player shadows. Documentary style.",
        "Ballet rehearsal studio, dancers at barre in different positions, mirrors reflecting multiple layers, natural light through high windows. Fine art photography.",
        "Traditional tea ceremony gathering, people kneeling at different heights on tatami, steam rising from tea bowls, soft natural lighting through paper screens. Cultural photography.",
        "Murmuration of starlings at dusk, birds forming fluid patterns against sunset sky, silhouetted trees in foreground. Nature photography.",
        "Penguin colony on Antarctic ice shelf, birds clustered at different levels, some diving into water, others tobogganing down slopes. Wildlife photography.",
        "Jellyfish aquarium display, creatures floating at different depths, bioluminescent glow, subtle blue lighting creating layers. Underwater photography.",
        "Outdoor flower market morning rush, vendors arranging bouquets in tiered displays, customers browsing, morning mist diffusing sunlight. Street photography.",
        "Orchestra during performance, musicians arranged in traditional sections, conductor raised, instruments catching stage lights. Concert photography.",
        "Beekeeper inspecting stacked hive boxes, bees swarming in sunlight, smoke creating atmospheric layers. Macro photography mixed with wide shots.",
    ]
    args = []
    for prompt in refined_prompts + refined_prompts_2:
        url = URLS[random.randint(0, len(URLS) - 1)]
        args.append((prompt, url))

    i = 0
    for image_bytes in generate_image.starmap(args):
        prompt = args[i][0].replace(" ", "_")
        i += 1

        v = 1 # increment on each run to avoid collisions
        image_path = AESTHETICS_SET / f"{prompt}_v{v}.png"

        with open(image_path, "wb") as f:
            f.write(image_bytes)

# we need to mount in our classifier library
mount = modal.Mount.from_local_dir(
    local_path="evals/improved_aesthetic_predictor", 
    remote_path="/root/improved_aesthetic_predictor"
)
@app.cls(image=image, mounts=[mount])
class ImprovedAestheticPredictor:
    def __init__(self):
        pass

    @modal.build()
    def download(self):
        import os
        from improved_aesthetic_predictor import download_models
        download_models()

    @modal.enter()
    def load(self):    
        from improved_aesthetic_predictor.model import load_models, predict
        model, clip, preprocess = load_models()
        self.model = model
        self.clip = clip
        self.preprocess = preprocess
        self.predict = predict

    @modal.method()
    def score(self, image_bytes):
        # lazy way to convert bytes to opencv image by writing to tmp file
        # we allow concurrent inputs, so use uuid for the file to avoid collisions
        try:
            uuid = str(uuid4())
            tmp_path = f"tmp_{uuid}.png"
            with open(tmp_path, "wb") as f:
                f.write(image_bytes)

            score = self.predict(self.model, self.clip, self.preprocess, tmp_path)
            print("Score", score)

        except Exception as e:
            raise e
        finally:
            os.remove(tmp_path)

@app.local_entrypoint()
def compare_aesthetics_predictors():
    good_dir = AESTHETICS_SET / "good"
    bad_dir = AESTHETICS_SET / "bad"
    good_imgs = list(good_dir.glob("*.png"))
    bad_imgs = list(bad_dir.glob("*.png"))

    predictor = ImprovedAestheticPredictor()

    good_img_bytes = [img.read_bytes() for img in good_imgs]
    bad_img_bytes = [img.read_bytes() for img in bad_imgs]
    
    for score in predictor.score.map(good_img_bytes[:1]):
        print(score)


# TODO make this run: modal run eval_aesthetics::compare_aesthetics_predictors