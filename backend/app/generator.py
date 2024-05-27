"""Text-conditioned generative model of QR code images."""
from dataclasses import dataclass

import modal

from .common import app

inference_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "accelerate",
        "datasets",
        "diffusers",
        "Pillow",
        "torch",
        "transformers",
        "triton",
        "xformers",
    )
    .apt_install("ffmpeg", "libsm6", "libxext6")
)


@dataclass
class InferenceConfig:
    """Configuration information for inference."""

    num_inference_steps: int = 50
    controlnet_conditioning_scale = [0.45, 0.25]
    guidance_scale: int = 9


CONFIG = InferenceConfig()


@app.cls(
    image=inference_image,
    gpu="a10g",
    secrets=[modal.Secret.from_name("huggingface")],
    keep_warm=1,
    container_idle_timeout=1200,
    allow_concurrent_inputs=10,
)
class Model:
    def setup(self, with_cuda=False):
        import os

        import huggingface_hub
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
        import torch

        hf_key = os.environ["HUGGINGFACE_TOKEN"]
        huggingface_hub.login(hf_key)

        brightness_controlnet = ControlNetModel.from_pretrained(
            "ioclab/control_v1p_sd15_brightness", torch_dtype=torch.float16
        )

        tile_controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1e_sd15_tile",
            torch_dtype=torch.float16,
            use_safetensors=False,
        )

        controller = StableDiffusionControlNetPipeline.from_pretrained(
            "Lykon/DreamShaper",
            controlnet=[brightness_controlnet, tile_controlnet],
            torch_dtype=torch.float16,
            use_safetensors=False,
            safety_checker=None,
        )

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
        input_image = input_image.resize((512, 512), resample=PIL.Image.LANCZOS)
        tile_input_image = self.resize_for_condition_image(input_image)
        output_image = self.pipe(
            text,
            image=[input_image, tile_input_image],
            height=512,
            width=512,
            num_inference_steps=CONFIG.num_inference_steps,
            controlnet_conditioning_scale=CONFIG.controlnet_conditioning_scale,
            guidance_scale=CONFIG.guidance_scale,
        )["images"][0]

        # blend the input QR code with the output image to improve scanability
        output_image = PIL.Image.blend(input_image, output_image, 0.9)

        buffer = io.BytesIO()
        output_image.save(buffer, format="PNG")
        buffer.seek(0)
        png_bytes = buffer.getvalue()

        return png_bytes
