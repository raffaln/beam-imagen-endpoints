"""
Beam Endpoint: DreamShaper XL (Lykon/dreamshaper-xl-1-0)
Calidad moderna y composición avanzada basada en la arquitectura SDXL.
"""

import time
import base64
import random
from io import BytesIO
from beam import endpoint, Image

CACHE_PATH = "/weights"


def load_model():
    import torch
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "Lykon/dreamshaper-xl-1-0",
        torch_dtype=torch.float16,
        cache_dir=CACHE_PATH,
        local_files_only=True
    )
    pipe.to("cuda")
    return pipe


@endpoint(
    name="beam-dreamshaper-xl",
    on_start=load_model,
    gpu="A10G",
    cpu=2,
    memory="24Gi",
    keep_warm_seconds=60,
    image=Image(
        python_version="python3.10",
        python_packages=[
            "torch==2.1.2",
            "torchvision==0.16.2",
            "diffusers==0.27.2",
            "transformers==4.38.2",
            "accelerate==0.29.3",
            "huggingface_hub==0.25.2",
            "pillow",
            "sentencepiece",
            "numpy<2",
        ],
        commands=[
            "mkdir -p /weights",
            "python3 -c 'import torch; from diffusers import StableDiffusionXLPipeline; StableDiffusionXLPipeline.from_pretrained(\"Lykon/dreamshaper-xl-1-0\", torch_dtype=torch.float16, cache_dir=\"/weights\")'"
        ]
    ),
)
def generate(context, **inputs):
    import torch

    pipe = context.on_start_value

    prompt = inputs.get("prompt", "")
    negative_prompt = inputs.get("negative_prompt", "")
    width = int(inputs.get("width", 1024))
    height = int(inputs.get("height", 1024))
    steps = int(inputs.get("steps", 30))
    guidance = inputs.get("guidance")
    seed = inputs.get("seed")
    image_base64 = inputs.get("image_base64")
    strength = inputs.get("strength")

    if strength is not None:
        strength = float(strength)
    else:
        strength = 0.55

    if seed is None or seed < 0:
        seed = random.randint(0, 2147483647)
    else:
        seed = int(seed)

    if guidance is None:
        guidance = 6.0
    else:
        guidance = float(guidance)

    start_time = time.time()
    try:
        generator = torch.Generator("cuda").manual_seed(seed)
        
        if image_base64:
            from diffusers import StableDiffusionXLImg2ImgPipeline
            from PIL import Image as PILImage
            import base64
            from io import BytesIO
            
            if "base64," in image_base64:
                image_base64 = image_base64.split("base64,")[1]
            init_image = PILImage.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")
            init_image = init_image.resize((width, height))
            
            img2img_pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)
            
            pipe_kwargs = {
                "prompt": prompt,
                "image": init_image,
                "strength": strength,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "generator": generator,
            }
            if negative_prompt:
                pipe_kwargs["negative_prompt"] = negative_prompt
                
            image = img2img_pipe(**pipe_kwargs).images[0]
        else:
            pipe_kwargs = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "generator": generator,
            }
            if negative_prompt:
                pipe_kwargs["negative_prompt"] = negative_prompt

            image = pipe(**pipe_kwargs).images[0]

        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "success": True,
            "model_name": "dreamshaper-xl",
            "image_base64": img_str,
            "seed": seed,
            "width": width,
            "height": height,
            "processing_time_seconds": round(time.time() - start_time, 2),
        }
    except Exception as e:
        return {"success": False, "model_name": "dreamshaper-xl", "error": str(e)}
