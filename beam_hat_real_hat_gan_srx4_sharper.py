"""
Beam Endpoint: Real-HAT-GAN SRx4 Sharper
Superresolución premium 4x con máximo detalle.
"""

import os
import time
import base64
import urllib.request
from io import BytesIO
from beam import endpoint, Image as BeamImage, Volume

CACHE_PATH = "./weights"
MODEL_URL = "https://huggingface.co/Acly/hat/resolve/main/Real_HAT_GAN_SRx4_sharper.pth"
MODEL_FILE = "/weights/Real_HAT_GAN_SRx4_sharper.pth"


def load_model():
    import torch
    import spandrel_extra_arches
    spandrel_extra_arches.install()

    from spandrel import ModelLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelLoader(device=device).load_from_file(MODEL_FILE)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model


@endpoint(
    name="beam-hat-real-hat-gan-srx4-sharper",
    on_start=load_model,
    gpu="A10G",
    cpu=2,
    memory="16Gi",
    keep_warm_seconds=60,
    image=BeamImage(
        python_version="python3.10",
        python_packages=[
            "torch", "torchvision", "spandrel", "spandrel_extra_arches", "pillow", "numpy"
        ],
        commands=[
            "mkdir -p /weights",
            "python3 -c 'import urllib.request; urllib.request.urlretrieve(\"https://huggingface.co/Acly/hat/resolve/main/Real_HAT_GAN_SRx4_sharper.pth\", \"/weights/Real_HAT_GAN_SRx4_sharper.pth\")'"
        ]
    ),
)
def upscale(context, **inputs):
    import torch
    import numpy as np
    from PIL import Image as PILImage

    model = context.on_start_value
    image_base64 = inputs.get("image_base64", "")
    upscale_factor = int(inputs.get("upscale_factor", 4))

    if not image_base64:
        return {"success": False, "model_name": "hat-real-hat-gan-srx4-sharper", "error": "No image_base64 provided."}

    start_time = time.time()
    try:
        if "base64," in image_base64:
            image_base64 = image_base64.split("base64,")[1]

        raw_bytes = base64.b64decode(image_base64)
        input_image = PILImage.open(BytesIO(raw_bytes)).convert("RGB")

        img_tensor = torch.from_numpy(np.array(input_image)).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(model.device)

        with torch.no_grad():
            output_tensor = model(img_tensor)

        output_arr = output_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        output_arr = (output_arr * 255).round().astype(np.uint8)
        output_image = PILImage.fromarray(output_arr)

        if upscale_factor != 4:
            w, h = input_image.size
            output_image = output_image.resize(
                (w * upscale_factor, h * upscale_factor), PILImage.Resampling.LANCZOS
            )

        buffered = BytesIO()
        output_image.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "success": True,
            "model_name": "hat-real-hat-gan-srx4-sharper",
            "image_base64": img_str,
            "width": output_image.width,
            "height": output_image.height,
            "processing_time_seconds": round(time.time() - start_time, 2),
        }
    except Exception as e:
        return {"success": False, "model_name": "hat-real-hat-gan-srx4-sharper", "error": str(e)}
