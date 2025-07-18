"""Text-conditioned generative model of aesthetically pleasing corrupt QR codes."""

import base64
import functools
import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import modal

volume = modal.Volume.from_name("qart-models", create_if_missing=True)
app = modal.App(name="qart-inference")

VOLUME_PATH = Path("/vol")
MODELS_PATH = VOLUME_PATH / "models"

here = Path(__file__).parent

inference_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsm6", "libxext6")
    .pip_install(
        "accelerate~=1.8.1",
        "datasets~=3.6.0",
        "diffusers==0.33.1",
        "Pillow~=11.2.1",
        "torch==2.7.1",
        "transformers==4.52.4",
        "triton==3.3.1",
        "huggingface-hub==0.33.0",
        "hf-transfer==0.1.9",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_CACHE": str(MODELS_PATH),
        }
    )
)

with inference_image.imports():
    import torch
    from diffusers import (
        ControlNetModel,
        DDIMScheduler,
        StableDiffusionControlNetPipeline,
    )
    from PIL import Image


@dataclass
class InferenceConfig:
    """Configuration information for inference."""

    num_inference_steps: int = 50
    controlnet_conditioning_scale: float = 1.5
    control_guidance_start: float = 0.25
    control_guidance_end: float = 1.0
    guidance_scale: float = 3.5
    negative_prompt: str = (
        "worst quality, low quality, ugly, disfigured, low quality, blurry"
    )
    height: int = 768
    width: int = 768
    num_images_per_prompt: int = 8


CONFIG = InferenceConfig()


@app.cls(
    image=inference_image,
    gpu=["B200", "H100!", "H200"],
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={VOLUME_PATH: volume},
    min_containers=1,
    scaledown_window=1200,
)
@modal.concurrent(max_inputs=10)
class Model:
    def setup(self, with_cuda=False):
        controlnet = ControlNetModel.from_pretrained(
            "monster-labs/control_v1p_sd15_qrcode_monster",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            subfolder="v2",
        )

        controller = StableDiffusionControlNetPipeline.from_pretrained(
            "Lykon/DreamShaper",
            controlnet=controlnet,
            torch_dtype=torch.bfloat16,
            use_safetensors=False,
            safety_checker=None,
        )

        controller.scheduler = DDIMScheduler.from_config(controller.scheduler.config)

        # small optimizations
        controller.fuse_qkv_projections()
        controller.unet.to(memory_format=torch.channels_last)
        controller.vae.to(memory_format=torch.channels_last)

        controller = controller.to("cuda") if with_cuda else controller

        # the first cut is the deepest
        controller(
            "dummy prompt",
            image=Image.new("RGB", (CONFIG.width, CONFIG.height)),
            height=CONFIG.height,
            width=CONFIG.width,
            num_images_per_prompt=CONFIG.num_images_per_prompt,
            num_inference_steps=1,
        )

        return controller

    @modal.enter()
    def start(self):
        controller = self.setup(with_cuda=True)

        self.pipe = controller

    @modal.method()
    def generate(
        self,
        text,
        input_image,
        negative_prompt=CONFIG.negative_prompt,
        height=CONFIG.height,
        width=CONFIG.width,
        num_inference_steps=CONFIG.num_inference_steps,
        num_images_per_prompt=CONFIG.num_images_per_prompt,
        guidance_scale=CONFIG.guidance_scale,
        controlnet_conditioning_scale=CONFIG.controlnet_conditioning_scale,
        control_guidance_start=CONFIG.control_guidance_start,
        control_guidance_end=CONFIG.control_guidance_end,
        **kwargs,
    ):
        print(input_image[:10])
        print(input_image[-10:])

        if "base64," in input_image:
            input_image = input_image.split("base64,")[1]

        input_image = base64.b64decode(input_image)
        input_image = Image.open(io.BytesIO(input_image)).convert("RGB")
        input_image = input_image.resize((width, height), resample=Image.LANCZOS)
        output_images = self.pipe(
            text,
            negative_prompt=negative_prompt,
            image=input_image,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            **kwargs,
        )["images"]

        process_func = functools.partial(postprocess_image, input_image=input_image)
        with ThreadPoolExecutor(max_workers=min(len(output_images), 8)) as executor:
            bytes_images = list(executor.map(process_func, output_images))

        return bytes_images


@app.local_entrypoint()  # for testing
def main(text: str = None):
    qr_dataurl = (here / "assets" / "qr-dataurl.txt").read_text()

    if text is None:
        text = "neon green prism, glowing, reflective, iridescent, metallic,"
        " rendered with blender, trending on artstation"

    images_bytes = Model().generate.remote(text=text, input_image=qr_dataurl)

    out_dir = here / "tests" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ii, image_bytes in enumerate(images_bytes):
        out_path = out_dir / f"{slugify(text)}-{str(ii).zfill(2)}.png"
        out_path.write_bytes(image_bytes)
        print(f"saved output to {out_path}")


def postprocess_image(output_image, input_image):
    # blend the input QR code with the output image to improve scanability
    blended_image = Image.blend(input_image, output_image, 0.95)

    buffer = io.BytesIO()
    blended_image.save(buffer, format="PNG", optimize=False, compress_level=1)
    # blended_image.save(buffer, format="JPEG", quality=95, optimize=False)
    buffer.seek(0)
    return buffer.getvalue()


def slugify(string):
    return (
        string.lower()
        .replace(" ", "-")
        .replace(",", "")
        .replace(".", "")
        .replace("/", "")
    )
