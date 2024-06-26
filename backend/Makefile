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

api: modal-auth ## deploy the generator and API on Modal
	modal deploy generator
		$(if $(filter prod, $(value ENV)),\
			modal deploy api, \
      modal serve api \
		)

tests: modal-auth dev-environment ## run the tests (you'll need to run modal serve too)
	pytest -rP

modal-auth: environment ## authenticates to Modal
	@$(if $(value MODAL_TOKEN_ID),, \
		$(error MODAL_TOKEN_ID is not set. Please set it before running this target.))
	@$(if $(value MODAL_TOKEN_SECRET),, \
		$(error MODAL_TOKEN_SECRET is not set. Please set it before running this target.))
	@modal token set --token-id $(MODAL_TOKEN_ID) --token-secret $(MODAL_TOKEN_SECRET)

modal-token: environment ## creates token ID and secret for authentication with modal
	modal token new
	@echo "###"
	@echo "# Copy the token info from the file mentioned above into a .env file"
	@echo "###"

environment: ## sets up the environment
	pip install -r requirements.txt

dev-environment: environment ## sets up the environment for development
	pip install -r requirements.dev.txt

banner: ## prints the logo in the terminal
	@cat assets/logo
