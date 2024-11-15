import modal
from pathlib import Path
from uuid import uuid4
import os

from .common import app, generate_image

AESTHETICS_SET = Path(__file__).parent /  "data" / "aesthetics"
os.makedirs(AESTHETICS_SET, exist_ok=True)
SCANNABILITY_SET = Path(__file__).parent /  "data" / "scannability"
os.makedirs(SCANNABILITY_SET, exist_ok=True)

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


################################################################################
# Official Aesthetics eval
# This section is the best eval class we landed on.
# See lower in this file for experiments run to choose this.
################################################################################

# Below is a rewrite of the improved aesthetic predictor model
# Credit https://huggingface.co/camenduru/improved-aesthetic-predictor/blob/main/simple_inference.py
def normalized(a, axis=-1, order=2):
    import numpy as np 
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

MODEL_PATH = "sac+logos+ava1-l14-linearMSE.pth"
MODEL_DIR = "/models"
MODEL_DOWNLOAD_URL = "https://huggingface.co/camenduru/improved-aesthetic-predictor/resolve/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"

def download_models():
    import os
    import clip

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_PATH)
    if not os.path.exists(model_path):
        print("Downloading model...")
        os.system(f"wget {MODEL_DOWNLOAD_URL} -O {model_path}")
        print("Done.")
    else:
        print("Model already exists.")

    # Clip
    clip.load("ViT-L/14", device="cpu") # caches model
    
def load_models():
    import torch
    import pytorch_lightning as pl
    import torch.nn as nn
    import torch.nn.functional as F
    import clip

    # if you changed the MLP architecture during training, change it also here:
    class MLP(pl.LightningModule):
        def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
            super().__init__()
            self.input_size = input_size
            self.xcol = xcol
            self.ycol = ycol
            self.layers = nn.Sequential(
                nn.Linear(self.input_size, 1024),
                #nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 128),
                #nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                #nn.ReLU(),
                nn.Dropout(0.1),

                nn.Linear(64, 16),
                #nn.ReLU(),

                nn.Linear(16, 1)
            )

        def forward(self, x):
            return self.layers(x)

        def training_step(self, batch, batch_idx):
                x = batch[self.xcol]
                y = batch[self.ycol].reshape(-1, 1)
                x_hat = self.layers(x)
                loss = F.mse_loss(x_hat, y)
                return loss
        
        def validation_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer

    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
    s = torch.load("/models/sac+logos+ava1-l14-linearMSE.pth", weights_only=True)   # load the model you trained previously or the model available in this repo
    model.load_state_dict(s)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # and the CLIP encoder
    clip, preprocess = clip.load("ViT-L/14", device=device)
    return model, clip, preprocess

def predict(model, clip, preprocess, img_path):
    import torch
    from PIL import Image
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pil_image = Image.open(img_path)
    image = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = clip.encode_image(image)

        im_emb_arr = normalized(image_features.cpu().detach().numpy() )
        prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
        prediction = prediction.cpu().detach().numpy()[0][0]

    return prediction

# Now we wrap that in a modal class for GPU hosting
@app.cls(image=image, gpu="any", allow_concurrent_inputs=10)
class ImprovedAestheticPredictor:
    def __init__(self):
        pass

    @modal.build()
    def download(self):
        download_models()

    @modal.enter()
    def load(self):    
        model, clip, preprocess = load_models()
        self.model = model
        self.clip = clip
        self.preprocess = preprocess
        self.predict = predict
        print("Aesthetics Predictor loaded.")

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
            return score

        except Exception as e:
            raise e
        finally:
            os.remove(tmp_path)

################################################################################
# Experiments while developing the aesthetics eval
# These are not used in the final eval, but are useful for showing the process
################################################################################

# from quart root:
# modal run backend.evals.aesthetics::copy_scannability_set
@app.local_entrypoint()
def copy_scannability_set():
    import shutil
    # First, copy in the sorted scannability se
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

# need multiple URLs so the idea of aesthetics doesn't 
# accidentally confuse a specific QR as aesthetic
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

# local entrypoint for generating ./data/aesthetics set
# from quart root:
# modal run backend.evals.aesthetics::generate_aesthetics_set_random
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

# local entrypoint for trying to generate more aesthetic images
# this was run multiple times
# from quart root:
# modal run backend.evals.aesthetics::generate_aesthetics_set_manual
@app.local_entrypoint()
def generate_aesthetics_set_manual():
    import random

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

@app.local_entrypoint()
def compare_aesthetics_predictions():
    good_dir = AESTHETICS_SET / "good"
    bad_dir = AESTHETICS_SET / "bad"
    if not good_dir.exists() or not bad_dir.exists():
        print("Please manually sort the images into good and bad buckets")
    good_imgs = list(good_dir.glob("*.png"))
    bad_imgs = list(bad_dir.glob("*.png"))

    predictor = ImprovedAestheticPredictor()

    good_img_bytes = [img.read_bytes() for img in good_imgs]
    bad_img_bytes = [img.read_bytes() for img in bad_imgs]
    
    print("Good images:")
    good_scores = []
    for score in predictor.score.map(good_img_bytes):
        print(score)
        good_scores.append(score)

    print("Bad images:")
    bad_scores = []
    for score in predictor.score.map(bad_img_bytes):
        print(score)
        bad_scores.append(score)
    
    from matplotlib import pyplot as plt

    # Create a histogram of the scores
    plt.figure(figsize=(10, 6))
    plt.hist([good_scores, bad_scores], 
            bins=50, 
            range=(2, 8), 
            label=['Good', 'Bad'],
            histtype='bar',
            align='mid')  # Align bars to the middle of bins

    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title('Distribution of Good vs Bad Scores')
    plt.legend()

    # Save to file
    plt.savefig("aesthetics_scores.png")
    plt.close()  # Close the figure to free memory
