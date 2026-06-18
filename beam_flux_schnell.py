"""
Beam serverless endpoint - FLUX.1-schnell GGUF Q4_K_S
~6.8GB download (GGUF transformer), ~7GB VRAM, 1-4 steps, excellent quality
Apache 2.0 license - commercial use allowed
"""

from beam import Image, Volume, endpoint, Output, env

if env.is_remote():
    from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
    import torch
    from huggingface_hub import hf_hub_download
    import os
    import uuid

image = (
    Image(python_version="python3.11")
    .add_python_packages(
        [
            "diffusers[torch]>=0.30",
            "transformers>=4.44",
            "huggingface_hub[hf-transfer]>=0.24",
            "torch>=2.3",
            "accelerate>=0.33",
            "safetensors>=0.4",
            "pillow>=10.0",
            "xformers>=0.0.27",
            "torchvision>=0.18",
            "gguf>=0.10.0",
        ]
    )
    .with_envs(
        "HF_HUB_ENABLE_HF_TRANSFER=1",
        "HF_HUB_DISABLE_PROGRESS_BARS=1",
    )
)

CACHE_PATH = "./models"
GGUF_REPO = "city96/FLUX.1-schnell-gguf"
GGUF_FILE = "flux1-schnell-Q4_K_S.gguf"
BASE_MODEL = "black-forest-labs/FLUX.1-schnell"


def load_models():
    ckpt_path = hf_hub_download(
        repo_id=GGUF_REPO,
        filename=GGUF_FILE,
        cache_dir=CACHE_PATH,
    )

    transformer = FluxTransformer2DModel.from_single_file(
        ckpt_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
        disable_mmap=True,
    )

    pipe = FluxPipeline.from_pretrained(
        BASE_MODEL,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_PATH,
    )

    pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing("max")

    return pipe


@endpoint(
    name="flux-schnell-gguf",
    image=image,
    on_start=load_models,
    keep_warm_seconds=300,
    cpu=2,
    memory="16Gi",
    gpu="A10G",
    volumes=[Volume(name="models", mount_path=CACHE_PATH)],
)
def generate(context, prompt=None, width=1024, height=1024, steps=4, guidance=3.5):
    if prompt is None:
        return {"error": "prompt is required"}

    pipe = context.on_start_value

    seed = context.get("seed", None)
    generator = None
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        output_type="pil",
    ).images[0]

    output = Output.from_pil_image(result).save()
    url = output.public_url()

    return {
        "image_url": url,
        "model": "FLUX.1-schnell GGUF Q4",
        "seed": seed,
    }