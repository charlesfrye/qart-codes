"""Text-conditioned generative model of QR code images."""
from dataclasses import dataclass

from modal import Image, Secret, SharedVolume, method

from .common import ROOT_DIR
from .common import stub

MODELS_DIR = ROOT_DIR / ".cache" / "huggingface"

model_volume = SharedVolume(cloud="aws").persist("qart-models-vol")

inference_image = (
    Image.debian_slim(python_version="3.10")
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

    num_inference_steps: int = 100
    controlnet_conditioning_scale = [0.45, 0.25]
    guidance_scale: int = 9


@stub.cls(
    image=inference_image,
    gpu="A10G",
    shared_volumes={str(MODELS_DIR): model_volume},
    secret=Secret.from_name("huggingface"),
    cloud="aws",
    keep_warm=1,
    container_idle_timeout=120,
)
class Model:
    config = InferenceConfig()

    def __enter__(self):
        import os

        from accelerate.utils import write_basic_config
        import huggingface_hub
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
        import torch

        write_basic_config(mixed_precision="fp16")

        hf_key = os.environ["HUGGINGFACE_TOKEN"]
        os.environ["HUGGINGFACE_CACHE"] = str(MODELS_DIR)
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
        ).to("cuda")

        controller.enable_xformers_memory_efficient_attention()

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

    @method()
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
            num_inference_steps=self.config.num_inference_steps,
            controlnet_conditioning_scale=self.config.controlnet_conditioning_scale,
            guidance_scale=self.config.guidance_scale,
        )["images"][0]

        # blend the input QR code with the output image to improve scanability
        output_image = PIL.Image.blend(input_image, output_image, 0.9)

        return output_image
