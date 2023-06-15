from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
import modal
from modal import Dict, Image, Mount, Secret, SharedVolume, Stub, asgi_app, method
from pydantic import BaseModel
import toml

# Specify the path to your pyproject.toml file
toml_file_path = Path("pyproject.toml")

ROOT_DIR = Path("/") / "root"
ASSETS_DIR = ROOT_DIR / "assets"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / ".cache" / "huggingface"

model_volume = SharedVolume(cloud="aws").persist("qart-models-vol")
results_volume = SharedVolume().persist("qart-results-vol")


toml_file_mount = Mount.from_local_file(
    local_path=toml_file_path, remote_path=ROOT_DIR / toml_file_path
)

assets_mount = Mount.from_local_dir(local_path=Path("assets"), remote_path=ASSETS_DIR)

# Read and parse the toml file
with open(toml_file_path, "r") as toml_file:
    pyproject = toml.load(toml_file)

info = pyproject["tool"]["poetry"]

api_image = Image.debian_slim().pip_install("wonderwords", "Pillow")

stub = Stub("qart", image=api_image, mounts=[toml_file_mount, assets_mount])


class HealthResponse(BaseModel):
    status: str
    status_unserious: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "status": "200 OK",
            }
        }


class HealthRequest(BaseModel):
    serious: bool = True


class JobStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


if modal.is_local:
    stub.jobs = Dict({"_test": {"status": JobStatus.COMPLETE}})
    with open(Path("assets") / "qr-dataurl.txt") as f:
        test_qr_dataurl = f.read()
else:
    with open(ASSETS_DIR / "qr-dataurl.txt") as f:
        test_qr_dataurl = f.read()


class ImagePayload(BaseModel):
    image_data: str


class JobRequest(BaseModel):
    prompt: str
    image: ImagePayload

    class Config:
        schema_extra = {
            "example": {
                "prompt": "a Shiba Inu drinking an Americano and eating pancakes",
                "image": ImagePayload(image_data=test_qr_dataurl),
            }
        }


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus

    class Config:
        schema_extra = {"example": {"job_id": "_test", "status": JobStatus.COMPLETE}}


@stub.function(timeout=150, shared_volumes={RESULTS_DIR: results_volume}, keep_warm=1)
def generate_and_save(job_id: str, prompt: str, image: str):
    """Generate a QR code from a prompt and save it to a file."""
    try:
        path = _path_from_job_id(job_id)
        dir_path = path.parent
        dir_path.mkdir(parents=True, exist_ok=True)
        call = Model().generate.spawn(prompt, image)
        _set_status(job_id, JobStatus.RUNNING)
    except Exception as e:
        _set_status(job_id, JobStatus.FAILED)
        stub.app.jobs[job_id]["error"] = e
        return

    # await the result
    try:
        generator_response = call.get(timeout=60)
    except TimeoutError as e:
        _set_status(job_id, JobStatus.FAILED)
        stub.app.jobs[job_id]["error"] = e
        return

    # write the result to a file
    _save_qr_code(generator_response, path)

    _set_status(job_id, JobStatus.COMPLETE)


@stub.function(shared_volumes={RESULTS_DIR: results_volume}, keep_warm=10, container_idle_timeout=60)
@asgi_app()
def api():
    api_backend = FastAPI(
        title=info["name"],
        description=f"{info['description']}. See {info['homepage']} to try it out!",
        version=info["version"],
        docs_url="/",
    )

    api_backend.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # healthcheck route
    @api_backend.get(
        "/health",
        response_description="If you get this response, the server is healthy.",
        name="health-check",
        operation_id="health",
        response_model=HealthResponse,
        response_model_exclude_unset=True,
        response_model_exclude_none=True,
        tags=["internal"],
    )
    async def healthcheck(serious: bool = True) -> HealthResponse:
        """vibecheck"""
        response = {"status": "200 OK"}
        if not serious:
            response["status-unserious"] = "200 more like 💯 mirite"
        return HealthResponse(**response)

    @api_backend.post(
        "/job",
        response_description="Info on the job you submitted.",
        name="start-job",
        summary="Start a QR code generation job",
        operation_id="start-job",
        response_model=JobStatusResponse,
        tags=["jobs"],
    )
    async def start_job(request: JobRequest) -> JobStatusResponse:
        from wonderwords import RandomWord

        r_gen = RandomWord()
        verb = r_gen.word(
            include_parts_of_speech=["verb"], word_min_length=4, word_max_length=7
        )
        adjective = r_gen.word(
            include_parts_of_speech=["adjective"], word_min_length=4, word_max_length=7
        )
        noun = r_gen.word(
            include_parts_of_speech=["noun"], word_min_length=4, word_max_length=7
        )

        job_id = "-".join([verb, adjective, noun])
        _handle_job(job_id, request)

        return JobStatusResponse(job_id=job_id, status=JobStatus.PENDING)

    @api_backend.get(
        "/job",
        name="check job status",
        summary="Check the status of a QR code generation job",
        operation_id="check-job",
        response_model=JobStatusResponse,
        tags=["jobs"],
    )
    async def check_job(
        job_id: str,
    ):
        status = _get_status(job_id)
        return JobStatusResponse(job_id=job_id, status=status)

    @api_backend.delete(
        "/job/{job_id}",
        name="cancel job",
        summary="Cancel an inference job",
        tags=["jobs"],
    )
    async def cancel_job(job_id: str):
        try:
            stub.app.jobs.get(job_id)["handle"].cancel()
            _set_status(job_id, JobStatus.CANCELLED)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found")
        return Response(status_code=204)

    @api_backend.get(
        "/job/{job_id}",
        name="get job result",
        summary="Get back the result of a QR code generation job",
        tags=["jobs"],
    )
    async def read_job(job_id) -> FileResponse:
        try:
            path = _path_from_job_id(job_id)
            if not path.exists():
                raise FileNotFoundError
            return FileResponse(path)
        except FileNotFoundError:
            return HTTPException(status_code=404, detail="Job result not found")

    return api_backend


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
        input_image = PIL.Image.open(io.BytesIO(base64.b64decode(input_image)))
        input_image = input_image.resize((512, 512), resample=PIL.Image.LANCZOS)
        tile_input_image = self.resize_for_condition_image(input_image)
        output_image = self.pipe(
            text,
            image=[input_image, tile_input_image],
            num_inference_steps=self.config.num_inference_steps,
            controlnet_conditioning_scale=self.config.controlnet_conditioning_scale,
            guidance_scale=self.config.guidance_scale,
        )["images"][0]

        return output_image


def _get_status(job_id: str) -> JobStatus:
    try:
        state = stub.app.jobs.get(job_id)
        return state["status"]
    except KeyError:
        print(job_id)
        return JobStatus.NOT_STARTED


def _set_status(job_id: str, status: JobStatus):
    print(f"{job_id} setting status to {status}")
    state = stub.app.jobs.pop(job_id)
    state["status"] = status
    stub.app.jobs.put(job_id, state)


def _path_from_job_id(job_id: str) -> Path:
    if job_id == "_test":
        return ASSETS_DIR / "qart.png"
    else:
        parts = job_id.split("-")
        result_path = Path(RESULTS_DIR, *parts) / "qr.png"
    return result_path


def _handle_job(job_id: str, request: JobRequest):
    try:
        if job_id == "_test":
            return
        else:
            stub.app.jobs.put(job_id, {"status": JobStatus.PENDING, "handle": None})
            call = generate_and_save.spawn(
                job_id, request.prompt, request.image.image_data
            )
            stub.app.jobs.put(job_id, {"status": JobStatus.PENDING, "handle": call})

    except Exception as e:
        print(e)
        print(f"setting status of {job_id} to failed")
        _set_status(job_id, JobStatus.FAILED)


def _save_qr_code(image, path):
    print(path)
    return image.save(path)
