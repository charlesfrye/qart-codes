## Getting Started

- Run `make` for a list of commands
- Run `make dev-environment` inside a Python virtual environment with version given by `.python-version`
- Add the Modal token ID and secret to `.env` and `.env.dev`.
- Run `make tests` to test each of the components (see the Makefile for testing commands)
- Run `make backend` to deploy the image generator service and inference-time evals
- Run `make api` to serve the jobs API around the image generator and evals
- If testing, set the `DEV_BACKEND_URL` in `.env` to the printed Modal URL
- Run `make e2etests` to do an e2e test for the whole backend
- Run `ENV=prod make api` to deploy the jobs API
