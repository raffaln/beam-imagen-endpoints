"""
Beam Endpoint: Qwen-Image-2512
Generación premium de imágenes de alta resolución.
"""

import time
import base64
from io import BytesIO
from beam import endpoint, Image, Volume

CACHE_PATH = "./weights"


def load_model():
    import torch
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(
        "Qwen/Qwen-Image-2512",
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_PATH,
    )
    pipe.to("cuda")
    return pipe


@endpoint(
    name="beam-qwen-image-2512",
    on_start=load_model,
    gpu="L40S",
    cpu=2,
    memory="24Gi",
    keep_warm_seconds=60,
    volumes=[Volume(name="weights", mount_path=CACHE_PATH)],
    image=Image(
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
)
def generate(**inputs):
    import torch
    import random

    pipe = generate.get_model()

    prompt = inputs.get("prompt", "")
    negative_prompt = inputs.get("negative_prompt", "")
    width = int(inputs.get("width", 1024))
    height = int(inputs.get("height", 1024))
    steps = int(inputs.get("steps", 20))
    guidance = inputs.get("guidance")
    seed = inputs.get("seed")

    if seed is None or seed < 0:
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
