import os
import time
import torch
import base64
import urllib.request
from io import BytesIO
from PIL import Image
import numpy as np
from beam import Image as BeamImage, endpoint, Volume

CACHE_PATH = "./weights"
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_FILE = os.path.join(CACHE_PATH, "RealESRGAN_x4plus.pth")

def load_model():
    """
    Descarga y carga los pesos de RealESRGAN_x4plus usando spandrel en on_start.
    """
    if not os.path.exists(MODEL_FILE):
        os.makedirs(CACHE_PATH, exist_ok=True)
        print("Descargando pesos de RealESRGAN_x4plus...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print("Descarga completa.")
        
    from spandrel import ModelLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_loader = ModelLoader(device=device)
    model = model_loader.load_from_file(MODEL_FILE)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model

@endpoint(
    on_start=load_model,
    gpu="T4",
    cpu=1,
    memory="16Gi",
    volumes=[Volume(name="weights", mount_path=CACHE_PATH)],
    image=BeamImage(
        python_version="python3.10",
        python_packages=["torch", "torchvision", "spandrel", "pillow", "numpy"],
    ),
)
def upscale(context, **kwargs):
    """
    Handler principal para procesar solicitudes de reescalado con Real-ESRGAN.
    """
    model = context.on_start_value
    
    image_base64 = kwargs.get("image_base64", "")
    upscale_factor = int(kwargs.get("upscale_factor", 4))
    
    if not image_base64:
        return {
            "success": False,
            "model_name": "real-esrgan-x4plus",
            "beam_endpoint_name": "beam-real-esrgan-x4plus",
            "error": "No image_base64 provided."
        }
        
    start_time = time.time()
    try:
        # Limpiar prefijo data URI si existe
        if "base64," in image_base64:
            image_base64 = image_base64.split("base64,")[1]
            
        raw_bytes = base64.b64decode(image_base64)
        input_image = Image.open(BytesIO(raw_bytes)).convert("RGB")
        
        # Convertir imagen PIL a tensor (normalizado 0-1)
        img_tensor = torch.from_numpy(np.array(input_image)).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(model.device)
        
        # Inferencia
        with torch.no_grad():
            output_tensor = model(img_tensor)
            
        # Convertir tensor resultante de vuelta a PIL
        output_arr = output_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        output_arr = (output_arr * 255).round().astype(np.uint8)
        output_image = Image.fromarray(output_arr)
        
        # Si el factor de escala pedido no es 4, redimensionamos usando Pillow LANCZOS
        if upscale_factor != 4:
            w, h = input_image.size
            output_image = output_image.resize((w * upscale_factor, h * upscale_factor), Image.Resampling.LANCZOS)
            
        # Codificar a base64
        buffered = BytesIO()
        output_image.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "model_name": "real-esrgan-x4plus",
            "beam_endpoint_name": "beam-real-esrgan-x4plus",
            "image_base64": img_str,
            "width": output_image.width,
            "height": output_image.height,
            "processing_time_seconds": round(processing_time, 2)
        }
    except Exception as e:
        return {
            "success": False,
            "model_name": "real-esrgan-x4plus",
            "beam_endpoint_name": "beam-real-esrgan-x4plus",
            "error": str(e)
        }
