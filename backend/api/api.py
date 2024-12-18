"""FastAPI interface for a pollable job queue around the generative model."""

from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .datamodel import JobStatus, JobRequest, JobStatusResponse, HealthResponse
from . import jobs


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
            response["status_unserious"] = "200 more like 💯 mirite"
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
        job_id = generate_job_id()
        jobs.start(job_id, request)

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
        status = jobs.check(job_id)
        return JobStatusResponse(job_id=job_id, status=status)

    @api_backend.delete(
        "/job/{job_id}",
        name="cancel job",
        summary="Cancel an inference job",
        tags=["jobs"],
    )
    async def cancel_job(job_id: str):
        try:
            jobs.cancel(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        return Response(status_code=204)

    @api_backend.get(
        "/job/{job_id}",
        name="get job result",
        summary="Get back the result of a QR code generation job",
        tags=["jobs"],
    )
    async def read_job(job_id) -> Response:
        try:
            result_bytes = jobs.read(job_id)

            return Response(result_bytes, media_type="image/png")
        except Exception as e:
            print(e)
            raise HTTPException(status_code=404, detail="Job result not found")

    return api_backend


def generate_job_id():
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

    return "-".join([verb, adjective, noun])
