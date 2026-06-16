"""
Beam Endpoint: Qwen-Image-2512
Generación premium de imágenes de alta resolución.
SDK: beam-sdk v0.15 (App + Runtime + rest_api)
"""

import time
import torch
import base64
from io import BytesIO
from beam import App, Runtime, Image as BeamImage, Volume

CACHE_PATH = "./weights"

app = App(
    name="beam-qwen-image-2512",
    runtime=Runtime(
        cpu=2,
        gpu="L40S",
        memory="24Gi",
        image=BeamImage(
            python_version="python3.10",
            python_packages=[
                "torch==2.1.2",
                "diffusers>=0.27.0",
                "transformers>=4.38.0",
                "accelerate>=0.27.0",
                "pillow",
                "sentencepiece",
            ],
        ),
    ),
    volumes=[Volume(name="weights", path=CACHE_PATH)],
)


def load_model():
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(
        "Qwen/Qwen-Image-2512",
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_PATH,
    )
    pipe.to("cuda")
    return pipe


@app.rest_api(loader=load_model)
def generate(**inputs):
    """
    Genera una imagen premium con Qwen-Image-2512.

    Inputs (JSON):
      - prompt (str): descripción de la imagen
      - negative_prompt (str, optional)
      - width (int, default 1024)
      - height (int, default 1024)
      - steps (int, default 20)
      - guidance (float, optional)
      - seed (int, optional)
    """
    pipe = generate.get_model()

    prompt = inputs.get("prompt", "")
    negative_prompt = inputs.get("negative_prompt", "")
    width = int(inputs.get("width", 1024))
    height = int(inputs.get("height", 1024))
    steps = int(inputs.get("steps", 20))
    guidance = inputs.get("guidance")
    seed = inputs.get("seed")

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
            "generator": generator,
        }
        if negative_prompt:
            pipe_kwargs["negative_prompt"] = negative_prompt
        if guidance is not None:
            pipe_kwargs["guidance_scale"] = float(guidance)

        image = pipe(**pipe_kwargs).images[0]

        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "success": True,
            "model_name": "qwen-image-2512",
            "image_base64": img_str,
            "seed": seed,
            "width": width,
            "height": height,
            "processing_time_seconds": round(time.time() - start_time, 2),
        }
    except Exception as e:
        return {"success": False, "model_name": "qwen-image-2512", "error": str(e)}
