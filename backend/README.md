## Getting Started

- Run `make` for a list of commands
- Run `make dev-environment` inside a Python virtual environment with version given by `.python-version`
- Add the Modal token ID and secret to `.env` and `.env.dev`.
- Run `modal run generator` to test the image generator service
- Run `modal deploy generator` to deploy the image generator service
- Run `modal serve api` to serve the backend API
- Run `ENV=dev make tests` to run an e2e test against the backend API and deployed image generator service
