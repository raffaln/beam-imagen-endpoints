"""
Beam Endpoint: Real-HAT-GAN SRx4 Sharper
Superresolución premium 4x con máximo detalle.
"""

import os
import time
import base64
import traceback
from io import BytesIO
from beam import endpoint, Image as BeamImage, Volume

CACHE_PATH = "./weights"
MODEL_REPO = "Acly/hat"
MODEL_FILE = "Real_HAT_GAN_sharper.pth"


def load_model():
    # Asegurar que la carpeta cache exista para los logs
    os.makedirs(CACHE_PATH, exist_ok=True)
    
    try:
        with open(os.path.join(CACHE_PATH, "startup.log"), "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] load_model started\n")
            
        import torch
        import spandrel_extra_arches
        
        with open(os.path.join(CACHE_PATH, "startup.log"), "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] spandrel_extra_arches imported. Installing extra arches...\n")
            
        spandrel_extra_arches.install()
        
        with open(os.path.join(CACHE_PATH, "startup.log"), "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] spandrel_extra_arches.install() completed\n")
            
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            cache_dir=CACHE_PATH,
        )
        
        with open(os.path.join(CACHE_PATH, "startup.log"), "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model weights resolved at {model_path}\n")

        from spandrel import ModelLoader
        
        # Verificar estado de CUDA
        cuda_ok = torch.cuda.is_available()
        device_str = "cuda" if cuda_ok else "cpu"
        device = torch.device(device_str)
        
        with open(os.path.join(CACHE_PATH, "startup.log"), "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] CUDA Available: {cuda_ok}, using device: {device_str}\n")
            
        model = ModelLoader(device=device).load_from_file(model_path)
        model.eval()
        if cuda_ok:
            model.cuda()
            
        with open(os.path.join(CACHE_PATH, "startup.log"), "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model loaded and compiled to GPU successfully\n")
            
        return model
    except Exception as e:
        err_msg = traceback.format_exc()
        with open(os.path.join(CACHE_PATH, "startup_error.log"), "w") as f:
            f.write(err_msg)
        with open(os.path.join(CACHE_PATH, "startup.log"), "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] CRASH: {str(e)}\n")
        raise e


@endpoint(
    name="beam-hat-real-hat-gan-srx4-sharper",
    on_start=load_model,
    gpu="A10G",
    cpu=2,
    memory="16Gi",
    keep_warm_seconds=60,
    volumes=[Volume(name="weights", mount_path=CACHE_PATH)],
    image=BeamImage(
        python_version="python3.10",
        python_packages=[
            "torch>=2.3,<2.6", "torchvision", "spandrel", "spandrel_extra_arches", "pillow", "numpy", "huggingface_hub"
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
