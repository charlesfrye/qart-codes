from modal import asgi_app
import toml

from .api import create as create_api
from .common import RESULTS_DIR, results_volume
from .common import stub
from .common import toml_file_path


@stub.function(
    network_file_systems={RESULTS_DIR: results_volume},
    keep_warm=10,
    container_idle_timeout=60,
)
@asgi_app()
def api():
    with open(toml_file_path, "r") as toml_file:
        pyproject = toml.load(toml_file)

    info = pyproject["tool"]["poetry"]

    api_backend = create_api(info)
    return api_backend
