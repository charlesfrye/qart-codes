"""Predict the human aesthetic score for an image.

This is a rewrite of the "improved aesthetic predictor" model, which combines CLIP embeddings and a small linear model.

For details see https://huggingface.co/camenduru/improved-aesthetic-predictor
"""
import io
import subprocess
from urllib.parse import quote
from pathlib import Path

import modal

app = modal.App(name="test-qart-aesthetics")

MODEL_FILE = "sac+logos+ava1-l14-linearMSE.pth"
REPO_URL = "https://huggingface.co/camenduru/improved-aesthetic-predictor"
MODEL_REVISION = "7b2449be1264fcd9a1cf92e3d30dd29af989c836"
MODEL_DOWNLOAD_URL = f"{REPO_URL}/resolve/{MODEL_REVISION}/{quote(MODEL_FILE)}"

MODEL_DIR = Path("/models")
MODEL_PATH = MODEL_DIR / MODEL_FILE

here = Path(__file__).parent
ASSETS_DIR = here.parent / "assets"


@app.local_entrypoint()  # for testing
def main(image_path: str = str(ASSETS_DIR / "qart.png"), target_score=4.72265625):
    result = Aesthetics().score.remote(Path(image_path).read_bytes())
    print(f"Aesthetic score: {result}")
    assert isclose(result, target_score)


def download_models(model_path=MODEL_PATH):
    import clip

    # load aesthetic predictor trained on CLIP embeddings
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        print("Downloading model...")
        subprocess.run(["wget", MODEL_DOWNLOAD_URL, "-O", model_path], check=True)
        print("Done.")

    # fix the format of the model -- it's multiple layers but can be collapsed to just one
    collapse_affine(model_path)

    # load CLIP
    clip.load("ViT-L/14")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget")
    .pip_install(
        "qrcode==8.2",
        "Pillow==11.2.1",
        "transformers==4.52.4",
        "torch==2.7.1",
        "git+https://github.com/openai/CLIP.git",
    )
    .run_function(download_models)
)

with image.imports():
    import clip
    import torch
    from PIL import Image

def normalize(a, axis=-1, order=2):
    l2 = torch.norm(a, p=order, dim=axis, keepdim=True)
    l2[l2 == 0] = 1
    return a / l2


def load_models():
    model = torch.nn.Linear(768, 1)  # embedding dim is 768 for CLIP ViT/L-14
    s = torch.load("/models/sac+logos+ava1-l14-linearMSE.pth", weights_only=True)
    model.load_state_dict(s)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    model.half()

    # and the CLIP encoder
    clip_model, preprocessor = clip.load("ViT-L/14", device=device)
    clip_model.eval()
    return model, clip_model, preprocessor


def predict(aesthetic_model, clip, preprocess, pil_image):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = clip.encode_image(image)

        normalized_image_features = normalize(image_features)
        prediction = aesthetic_model(normalized_image_features)

        prediction = prediction.cpu().numpy()[0][0]

    return prediction


# Now we wrap inference in a modal class for hosting
@app.cls(image=image, gpu="L40S", min_containers=1)
@modal.concurrent(max_inputs=10)
class Aesthetics:
    """Predict a human aesthetic ranking based on CLIP embeddings and a linear model."""

    @modal.enter()
    def load(self):
        self.aesthetic_model, self.clip, self.preprocessor = load_models()
        print("Aesthetics Predictor loaded.")

    @modal.method()
    def wake(self):
        pass

    @modal.method()
    def score(self, image_bytes: bytes) -> float:
        pil_image = Image.open(io.BytesIO(image_bytes))
        score = predict(self.aesthetic_model, self.clip, self.preprocessor, pil_image)
        return float(score)


def collapse_affine(model_path):
    """Overwrites a decomposed linear model with an equivalent composed linear model.

    The aesthetic model is actually a linear/affine model ('MLP with no non-linearities'), so we collapse it to one linear layer.

    That is, the code for the original torch.nn.Module is literally:

    ```python
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
    ```

    including the comments!
    """
    import torch

    state_dict = torch.load(
        model_path, weights_only=True, map_location=torch.device("cpu")
    )
    with torch.no_grad():
        W, b = tested_affine_composition(list(state_dict.values()))
        linear_model = make_torch_linear_from_weights(W, b)

    torch.save(linear_model.state_dict(), model_path)


def make_torch_linear_from_weights(weights, biases):
    import torch

    in_features = weights.shape[1]
    out_features = weights.shape[0]
    linear = torch.nn.Linear(in_features, out_features)

    with torch.no_grad():
        linear.weight.copy_(weights)
        linear.bias.copy_(biases)

    return linear


def compose_affine_transforms(params):
    """Given a sequence of weight and bias tensors, compute the overall affine transformation.

    Args:
        params (list of torch.Tensor): A sequence of tensors alternating between weight and bias.

    Returns:
        (torch.Tensor, torch.Tensor): The final weight matrix and bias vector.
    """
    assert len(params) % 2 == 0, "Provide alternating weight & bias tensors."

    if len(params) == 0:
        raise ValueError("Provide nonempty list of weight & bias tensors")

    # start with first transform
    W_total = params[0]
    b_total = params[1]

    # compose
    for i in range(2, len(params), 2):
        W, b = params[i], params[i + 1]

        W_total = W @ W_total  # apply linear term
        b_total = b_total @ W.T + b  # map bias

    return W_total, b_total


def tested_affine_composition(params, num_samples=1):
    """Composes affine transformations in params and

    Args:
        params (list of torch.Tensor): Alternating weight and bias tensors.
        input_dim (int): Dimensionality of the input space.
        num_samples (int): Number of random samples to test.

    Returns:
        torch.Tensor: The difference between sequential application and composed application.
    """
    import torch

    W_final, b_final = compose_affine_transforms(params)

    x = torch.randn(num_samples, W_final.shape[1])

    # apply provided transforms
    y_seq = x
    for i in range(0, len(params), 2):
        W, b = params[i], params[i + 1]
        y_seq = y_seq @ W.T + b  # PyTorch convention: xW^T + b

    # apply the composed transformation
    y_composed = x @ W_final.T + b_final

    # Compute the difference
    assert torch.allclose(y_seq, y_composed)

    return W_final, b_final


def isclose(a: float, b: float, atol: float = 1e-3, rtol: float = 5e-3) -> bool:
    """Pure Python version of isclose from numpy/torch"""
    return abs(a - b) <= atol + rtol * abs(b)
