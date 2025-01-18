import modal
# from ..generator import app, Model

image = (modal.Image.debian_slim()
    .pip_install("qrcode", "Pillow")
)

# shared function for generating qr code image
@app.function(
    image=image, 
)
def generate_image(prompt: str, qr_url: str) -> bytes:
    import base64
    import io
    import qrcode

    # produces pil image from url
    image = qrcode.make(qr_url) 
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    # put into base64 (format expected by modal)
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # generate image
    model = Model()
    image_bytes = model.generate.remote(prompt, image_base64)

    return image_bytes