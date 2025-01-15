import modal
from .aesthetics import app as aesthetics_app
from .scannability import app as scannability_app

app = modal.App(name="qart-eval")
app.include(aesthetics_app)
app.include(scannability_app)
