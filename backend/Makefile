# provide ENV=prod or ENV=dev
ENV_LOADED :=
ifeq ($(ENV), prod)
    ifneq (,$(wildcard ./.env))
        include .env
        export
				ENV_LOADED := Loaded config from .env
    endif
else
    ifneq (,$(wildcard ./.env.dev))
        include .env.dev
        export
				ENV_LOADED := Loaded config from .env.dev
    endif
endif

.PHONY: help
.DEFAULT_GOAL := help

help: banner ## get a list of all the targets, plus short descriptions
	@# source for the incantation: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | awk 'BEGIN {FS = ":.*?##"}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

app: modal-auth ## deploy the application on Modal
		$(if $(filter prod, $(value ENV)),\
			poetry run modal deploy app/main.py, \
      poetry run modal serve app/main.py \
		)

modal-auth: environment ## authenticates to Modal
	@$(if $(value MODAL_TOKEN_ID),, \
		$(error MODAL_TOKEN_ID is not set. Please set it before running this target.))
	@$(if $(value MODAL_TOKEN_SECRET),, \
		$(error MODAL_TOKEN_SECRET is not set. Please set it before running this target.))
	@poetry run modal token set --token-id $(MODAL_TOKEN_ID) --token-secret $(MODAL_TOKEN_SECRET)

modal-token: environment ## creates token ID and secret for authentication with modal
	poetry run modal token new
	@echo "###"
	@echo "# Copy the token info from the file mentioned above into a .env file"
	@echo "###"

environment: ## sets up the environment
	poetry install --without=dev

dev-environment: environment ## sets up the environment for development
	poetry install --sync

poetry-update: ## updates the poetry.lock file
	poetry update --lock

banner: ## prints the logo in the terminal
	@cat assets/logo
