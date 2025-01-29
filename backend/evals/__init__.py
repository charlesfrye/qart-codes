import modal
from .aesthetics import app as aesthetics_app
from .scannability_qreader import app as qreader_app

app = modal.App(name="qart-eval")

app.include(aesthetics_app)
app.include(qreader_app)
