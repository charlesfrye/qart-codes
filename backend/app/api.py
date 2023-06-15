from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
import modal
from modal import Dict

from .common import stub
from .common import RESULTS_DIR, results_volume
from .common import get_status, set_status, path_from_job_id
from .datamodel import JobStatus, JobRequest, JobStatusResponse, HealthResponse
from .generator import Model


def create(info) -> FastAPI:
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
        handle_job(job_id, request)

        return JobStatusResponse(job_id=job_id, status=JobStatus.PENDING)

    @api_backend.get(
        "/job",
        name="check job status",
        summary="Check the status of a QR code generation job",
        operation_id="check-job",
        response_model=JobStatusResponse,
        tags=["jobs"],
    )
    async def check_job(job_id: str):
        status = get_status(job_id)
        return JobStatusResponse(job_id=job_id, status=status)

    @api_backend.delete(
        "/job/{job_id}",
        name="cancel job",
        summary="Cancel an inference job",
        tags=["jobs"],
    )
    async def cancel_job(job_id: str):
        try:
            set_status(job_id, JobStatus.CANCELLED)
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
            path = path_from_job_id(job_id)
            if not path.exists():
                raise FileNotFoundError
            return FileResponse(path)
        except FileNotFoundError:
            return HTTPException(status_code=404, detail="Job result not found")

    return api_backend


def save_qr_code(image, path):
    print(path)
    return image.save(path)


if modal.is_local:
    stub.jobs = Dict({"_test": {"status": JobStatus.COMPLETE}})


@stub.function(timeout=150, shared_volumes={RESULTS_DIR: results_volume}, keep_warm=1)
def generate_and_save(job_id: str, prompt: str, image: str):
    """Generate a QR code from a prompt and save it to a file."""
    try:
        path = path_from_job_id(job_id)
        dir_path = path.parent
        dir_path.mkdir(parents=True, exist_ok=True)
        call = Model().generate.spawn(prompt, image)
        set_status(job_id, JobStatus.RUNNING)
    except Exception as e:
        set_status(job_id, JobStatus.FAILED)
        stub.app.jobs[job_id]["error"] = e
        return

    # await the result
    try:
        generator_response = call.get(timeout=60)
    except TimeoutError as e:
        set_status(job_id, JobStatus.FAILED)
        stub.app.jobs[job_id]["error"] = e
        return

    # write the result to a file
    save_qr_code(generator_response, path)

    set_status(job_id, JobStatus.COMPLETE)


def handle_job(job_id: str, request: JobRequest):
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
        set_status(job_id, JobStatus.FAILED)
