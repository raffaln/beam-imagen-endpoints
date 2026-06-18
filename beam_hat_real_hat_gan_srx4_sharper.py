"""
Beam Endpoint: Real-HAT-GAN SRx4 Sharper
Superresolución premium 4x con máximo detalle y soporte para mosaicos (tiling) anti-OOM.
"""

import os
# Configuración para evitar fragmentación de VRAM en PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import base64
import traceback
from io import BytesIO
from beam import endpoint, Image as BeamImage, Volume

CACHE_PATH = "./weights"
MODEL_REPO = "Acly/hat"
MODEL_FILE = "Real_HAT_GAN_sharper.pth"


def load_model():
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


def tiled_upscale(model, input_image, tile_size=256, tile_pad=16, scale=4):
    """
    Reescala la imagen por parches (tiles) para evitar errores de falta de memoria CUDA (OOM).
    """
    import torch
    import numpy as np
    from PIL import Image as PILImage
    
    w, h = input_image.size
    output_image = PILImage.new("RGB", (w * scale, h * scale))
    device = model.device
    
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            # Calcular límites del parche de entrada con relleno
            y1 = max(y - tile_pad, 0)
            x1 = max(x - tile_pad, 0)
            y2 = min(y + tile_size + tile_pad, h)
            x2 = min(x + tile_size + tile_pad, w)
            
            # Recortar parche
            tile = input_image.crop((x1, y1, x2, y2))
            
            # Inferencia del parche en float32
            img_tensor = torch.from_numpy(np.array(tile)).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            with torch.inference_mode():
                output_tensor = model(img_tensor)
                
            output_arr = output_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            output_arr = (output_arr * 255).round().astype(np.uint8)
            output_tile = PILImage.fromarray(output_arr)
            
            # Calcular la zona a recortar del parche de salida para quitar el relleno
            out_y_offset = (y - y1) * scale
            out_x_offset = (x - x1) * scale
            out_w = min(tile_size, w - x) * scale
            out_h = min(tile_size, h - y) * scale
            
            cropped_tile = output_tile.crop((
                out_x_offset,
                out_y_offset,
                out_x_offset + out_w,
                out_y_offset + out_h
            ))
            
            # Pegar el resultado en el lienzo final
            output_image.paste(cropped_tile, (x * scale, y * scale))
            
    # Limpiar memoria de CUDA al finalizar todo el reescalado
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
                
    return output_image


image_config = (
    BeamImage(python_version="python3.10")
    .add_python_packages([
        "torch>=2.3,<2.6", "torchvision", "spandrel", "spandrel_extra_arches", "pillow", "numpy", "huggingface_hub"
    ])
    .with_envs(
        "GUNICORN_CMD_ARGS=\"-t 600\"",
    )
)


@endpoint(
    name="beam-hat-real-hat-gan-srx4-sharper",
    on_start=load_model,
    gpu="A10G",
    cpu=2,
    memory="16Gi",
    keep_warm_seconds=60,
    volumes=[Volume(name="weights", mount_path=CACHE_PATH)],
    image=image_config,
)
def upscale(context, **inputs):
    import torch
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

        # Usar reescalado por parches (tiled_upscale) para evitar OOM
        output_image = tiled_upscale(model, input_image, tile_size=256, tile_pad=16, scale=4)

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
