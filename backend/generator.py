"""Text-conditioned generative model of QR code images."""

from dataclasses import dataclass
from pathlib import Path

import modal

app = modal.App(name="qart-inference")

inference_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "accelerate==0.21.0",
        "datasets==2.13.1",
        "diffusers==0.20.2",
        "Pillow~=10.0.0",
        "torch==2.0.1",
        "transformers==4.30.2",
        "triton==2.0.0",
        "xformers==0.0.20",
    )
    .apt_install("ffmpeg", "libsm6", "libxext6")
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
    gpu="a100",
    secrets=[modal.Secret.from_name("huggingface")],
    keep_warm=1,
    container_idle_timeout=1200,
    allow_concurrent_inputs=10,
)
class Model:
    def setup(self, with_cuda=False):
        import os

        import huggingface_hub
        from diffusers import (
            ControlNetModel,
            DDIMScheduler,
            StableDiffusionControlNetPipeline,
        )
        import torch

        hf_key = os.environ["HUGGINGFACE_TOKEN"]
        huggingface_hub.login(hf_key)

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

            controller.enable_xformers_memory_efficient_attention()

        return controller

    @modal.build()
    def download_models(self):
        self.setup(with_cuda=False)

    @modal.enter()
    def start(self):
        from accelerate.utils import write_basic_config

        write_basic_config()

        controller = self.setup(with_cuda=True)

        self.pipe = controller

    @staticmethod
    def resize_for_condition_image(input_image, resolution: int = 512):
        import PIL.Image

        input_image = input_image.convert("RGB")
        W, H = input_image.size
        k = float(resolution) / min(H, W)
        H *= k
        W *= k
        H = int(round(H / 64.0)) * 64
        W = int(round(W / 64.0)) * 64
        img = input_image.resize((W, H), resample=PIL.Image.LANCZOS)
        return img

    @modal.method()
    def generate(self, text, input_image):
        import base64
        import io

        import PIL.Image

        print(input_image[:10])
        print(input_image[-10:])
        if "base64," in input_image:
            input_image = input_image.split("base64,")[1]
        input_image = PIL.Image.open(io.BytesIO(base64.b64decode(input_image))).convert(
            "RGB"
        )
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
        text = "neon green prism, glowing, reflective, iridescent, metallic, rendered with blender, trending on artstation"

    image_bytes = Model.generate.remote(text=text, input_image=qr_dataurl)

    with open(
        Path(__file__).parent / "tests" / "out" / f"{slugify(text)}.png", "wb"
    ) as f:
        f.write(image_bytes)

    print("saved output to", f.name)


def slugify(string):
    return (
        string.lower()
        .replace(" ", "-")
        .replace(",", "")
        .replace(".", "")
        .replace("/", "")
    )
