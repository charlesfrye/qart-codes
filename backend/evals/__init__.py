import modal
from . import aesthetics
from . import scannability

app = modal.App(name="qart-eval")

app.include(aesthetics.app)
app.include(scannability.app)
