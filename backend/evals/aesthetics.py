import modal

app = modal.App(name="qart-aesthetics")

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
    clip.load("ViT-L/14", device="cpu")  # caches model


# we just share the same image across detectors
image = (
    modal.Image.debian_slim()
    .apt_install("git", "wget")
    .pip_install(
        "qrcode",
        "Pillow",
        "transformers",
        "torch",
        "pytorch-lightning",
        "git+https://github.com/openai/CLIP.git",
    )
    .run_function(download_models)
)


# Below is a rewrite of the improved aesthetic predictor model
# Credit https://huggingface.co/camenduru/improved-aesthetic-predictor/blob/main/simple_inference.py
def normalized(a, axis=-1, order=2):
    import numpy as np

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def load_models():
    import torch
    import pytorch_lightning as pl
    import torch.nn as nn
    import torch.nn.functional as F
    import clip

    # if you changed the MLP architecture during training, change it also here:
    class MLP(pl.LightningModule):
        def __init__(self, input_size, xcol="emb", ycol="avg_rating"):
            super().__init__()
            self.input_size = input_size
            self.xcol = xcol
            self.ycol = ycol
            self.layers = nn.Sequential(
                nn.Linear(self.input_size, 1024),
                # nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 128),
                # nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                # nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 16),
                # nn.ReLU(),
                nn.Linear(16, 1),
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
    s = torch.load(
        "/models/sac+logos+ava1-l14-linearMSE.pth", weights_only=True
    )  # load the model you trained previously or the model available in this repo
    model.load_state_dict(s)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # and the CLIP encoder
    clip, preprocess = clip.load("ViT-L/14", device=device)
    return model, clip, preprocess


def predict(model, clip, preprocess, pil_image):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = clip.encode_image(image)

        im_emb_arr = normalized(image_features.cpu().detach().numpy())
        prediction = model(
            torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor)
        )
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
    def wake(self):
        pass

    @modal.method()
    def score(self, image_bytes):
        import io
        from PIL import Image

        pil_image = Image.open(io.BytesIO(image_bytes))
        score = self.predict(self.model, self.clip, self.preprocess, pil_image)
        return float(score)
