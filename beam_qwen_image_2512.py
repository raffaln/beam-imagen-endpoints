import time
import torch
import base64
from io import BytesIO
from beam import Image, endpoint, Volume

CACHE_PATH = "./weights"

def load_model():
    """
    Carga el pipeline Qwen-Image-2512 al iniciar el contenedor.
    """
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(
        "Qwen/Qwen-Image-2512",
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_PATH
    )
    pipe.to("cuda")
    return pipe

@endpoint(
    on_start=load_model,
    gpu="L40S",
    cpu=2,
    memory="24Gi",
    volumes=[Volume(name="weights", mount_path=CACHE_PATH)],
    image=Image(
        python_version="python3.10",
        python_packages=["torch", "transformers", "accelerate", "diffusers", "pillow"],
    ),
)
def generate(context, **kwargs):
    """
    Handler principal para procesar solicitudes de generación premium con Qwen-Image-2512.
    """
    pipe = context.on_start_value
    
    prompt = kwargs.get("prompt", "")
    negative_prompt = kwargs.get("negative_prompt", "")
    width = int(kwargs.get("width", 1024))
    height = int(kwargs.get("height", 1024))
    steps = int(kwargs.get("steps", 20))
    guidance = kwargs.get("guidance")
    seed = kwargs.get("seed")
    
    if seed is None or seed < 0:
        import random
        seed = random.randint(0, 2147483647)
    else:
        seed = int(seed)
        
    start_time = time.time()
    try:
        generator = torch.Generator("cuda").manual_seed(seed)
        
        pipe_kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "generator": generator
        }
        if negative_prompt:
            pipe_kwargs["negative_prompt"] = negative_prompt
        if guidance is not None:
            pipe_kwargs["guidance_scale"] = float(guidance)
            
        # Inferencia
        image = pipe(**pipe_kwargs).images[0]
        
        # Conversión a base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "model_name": "qwen-image-2512",
            "beam_endpoint_name": "beam-qwen-image-2512",
            "image_base64": img_str,
            "seed": seed,
            "width": width,
            "height": height,
            "processing_time_seconds": round(processing_time, 2)
        }
    except Exception as e:
        return {
            "success": False,
            "model_name": "qwen-image-2512",
            "beam_endpoint_name": "beam-qwen-image-2512",
            "error": str(e)
        }
