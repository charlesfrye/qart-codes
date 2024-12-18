"""Text-conditioned generative model of QR code images."""

from dataclasses import dataclass
from pathlib import Path

import modal

volume = modal.Volume.from_name("qart-models", create_if_missing=True)
app = modal.App(name="qart-inference")

VOLUME_PATH = Path("/vol")
MODELS_PATH = VOLUME_PATH / "models"

inference_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsm6", "libxext6")
    .pip_install(
        "accelerate~=1.2.1",
        "datasets~=2.13.1",
        "diffusers==0.31.0",
        "Pillow~=10.0.0",
        "torch==2.5.1",
        "transformers==4.47.1",
        "triton==3.1.0",
        "huggingface-hub==0.27.0",
        "hf-transfer==0.1.8",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": str(MODELS_PATH)})
)


@dataclass
class InferenceConfig:
    """Configuration information for inference."""

    num_inference_steps: int = 100
    controlnet_conditioning_scale: float = 1.5
    guidance_scale: float = 8.0


CONFIG = InferenceConfig()


@app.cls(
    image=inference_image,
    gpu="h100",
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={VOLUME_PATH: volume},
    keep_warm=1,
    container_idle_timeout=1200,
    allow_concurrent_inputs=10,
)
class Model:
    def setup(self, with_cuda=False):
        from diffusers import (
            ControlNetModel,
            DDIMScheduler,
            StableDiffusionControlNetPipeline,
        )
        import torch

        controlnet = ControlNetModel.from_pretrained(
            "monster-labs/control_v1p_sd15_qrcode_monster",
            torch_dtype=torch.float16,
            use_safetensors=True,
            subfolder="v2",
        )

        controller = StableDiffusionControlNetPipeline.from_pretrained(
            "Lykon/DreamShaper",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            use_safetensors=False,
            safety_checker=None,
        )

        controller.scheduler = DDIMScheduler.from_config(controller.scheduler.config)

        if with_cuda:
            controller = controller.to("cuda") if with_cuda else controller

        return controller

    @modal.enter()
    def start(self):
        from accelerate.utils import write_basic_config

        write_basic_config(mixed_precision="fp16")

        controller = self.setup(with_cuda=True)

        self.pipe = controller

    @modal.method()
    def generate(self, text, input_image):
        import base64
        import io

        import PIL.Image

        print(input_image[:10])
        print(input_image[-10:])

        if "base64," in input_image:
            input_image = input_image.split("base64,")[1]

        input_image = base64.b64decode(input_image)
        input_image = PIL.Image.open(io.BytesIO(input_image)).convert("RGB")
        input_image = input_image.resize((768, 768), resample=PIL.Image.LANCZOS)

        output_image = self.pipe(
            text,
            negative_prompt="ugly, disfigured, low quality, blurry",
            image=input_image,
            height=768,
            width=768,
            num_inference_steps=CONFIG.num_inference_steps,
            controlnet_conditioning_scale=CONFIG.controlnet_conditioning_scale,
            guidance_scale=CONFIG.guidance_scale,
        )["images"][0]

        # blend the input QR code with the output image to improve scanability
        output_image = PIL.Image.blend(input_image, output_image, 0.85)

        buffer = io.BytesIO()
        output_image.save(buffer, format="PNG")
        buffer.seek(0)
        png_bytes = buffer.getvalue()

        return png_bytes


@app.local_entrypoint()
def main(text: str = None):
    qr_dataurl = (Path(__file__).parent / "assets" / "qr-dataurl.txt").read_text()

    if text is None:
        text = "neon green prism, glowing, reflective, iridescent, metallic,"
        " rendered with blender, trending on artstation"

    image_bytes = Model.generate.remote(text=text, input_image=qr_dataurl)

    out_path = Path(__file__).parent / "tests" / "out" / f"{slugify(text)}.png"
    out_path.write_bytes(image_bytes)

    print("saved output to", out_path)


def slugify(string):
    return (
        string.lower()
        .replace(" ", "-")
        .replace(",", "")
        .replace(".", "")
        .replace("/", "")
    )
