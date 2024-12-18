import modal
import toml

from .api import create as create_api
from .common import app
from .common import toml_file_path


@app.function(keep_warm=1, container_idle_timeout=60, allow_concurrent_inputs=1000)
@modal.asgi_app()
def api():
    with open(toml_file_path, "r") as toml_file:
        pyproject = toml.load(toml_file)

    info = pyproject["tool"]["poetry"]

    api_backend = create_api(info)
    return api_backend
