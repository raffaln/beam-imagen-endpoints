"""
Beam serverless endpoint - FLUX.1-schnell GGUF Q4_K_S
~6.8GB download (GGUF transformer), ~7GB VRAM, 1-4 steps, excellent quality
Apache 2.0 license - commercial use allowed
GPU: A10G (recommended) or RTX 4090
Guidance scale must be 0 for schnell (timestep-distilled model)
"""

from beam import Image, endpoint, Output, env

if env.is_remote():
    from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
    import torch
    from huggingface_hub import hf_hub_download, login
    import os
    import uuid

image = (
    Image(python_version="python3.11")
    .add_python_packages(
        [
            "diffusers[torch]>=0.30",
            "transformers>=4.44",
            "huggingface_hub[hf-transfer]>=0.24",
            "torch==2.5.1",
            "accelerate>=0.33",
            "safetensors>=0.4",
            "pillow>=10.0",
            "torchvision==0.20.1",
            "gguf>=0.10.0",
        ]
    )
    .with_envs({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    })
)

CACHE_PATH = "/tmp/models"
GGUF_REPO = "city96/FLUX.1-schnell-gguf"
GGUF_FILE = "flux1-schnell-Q4_K_S.gguf"
BASE_MODEL = "black-forest-labs/FLUX.1-schnell"


def load_models():
    # Debug: check CUDA and GPU availability
    print(f"DEBUG: torch version: {torch.__version__}")
    print(f"DEBUG: CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"DEBUG: device count: {torch.cuda.device_count()}")
        print(f"DEBUG: device 0 name: {torch.cuda.get_device_name(0)}")
        print(f"DEBUG: CUDA version: {torch.version.cuda}")
    else:
        print("DEBUG: CUDA NOT available - will try to force CPU fallback")
        # Try to import nvidia tools
        try:
            import subprocess
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
            print(f"DEBUG: nvidia-smi stdout: {result.stdout[:500]}")
            print(f"DEBUG: nvidia-smi stderr: {result.stderr[:500]}")
        except FileNotFoundError:
            print("DEBUG: nvidia-smi not found")
        except Exception as e:
            print(f"DEBUG: nvidia-smi error: {e}")

    # Authenticate with HuggingFace for gated model access
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    # Download GGUF to /tmp
    local_ckpt = hf_hub_download(
        repo_id=GGUF_REPO,
        filename=GGUF_FILE,
        cache_dir=CACHE_PATH,
    )

    transformer = FluxTransformer2DModel.from_single_file(
        local_ckpt,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
    )

    pipe = FluxPipeline.from_pretrained(
        BASE_MODEL,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_PATH,
    )

    # CRITICAL: force model to CUDA explicitly
    print("DEBUG: Moving pipe to CUDA...")
    pipe.to("cuda")
    print("DEBUG: Pipe on CUDA successfully")
    
    # Enable optimizations
    pipe.enable_attention_slicing("max")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    return pipe


@endpoint(
    name="flux-schnell-gguf",
    image=image,
    on_start=load_models,
    keep_warm_seconds=300,
    cpu=2,
    memory="16Gi",
    gpu="A10G",
)
def generate(context, prompt=None, width=1024, height=1024, steps=4, guidance=0.0, seed=None):
    if prompt is None:
        return {
            "success": False,
            "error": "prompt is required"
        }

    pipe = context.on_start_value

    generator = None
    if seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)

    try:
        result = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            output_type="pil",
        ).images[0]

        import base64
        from io import BytesIO
        
        buffered = BytesIO()
        result.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "success": True,
            "model_name": "flux-schnell-gguf",
            "beam_endpoint_name": "flux-schnell-gguf",
            "image_base64": img_str,
            "seed": seed,
            "width": width,
            "height": height
        }
    except Exception as e:
        return {
            "success": False,
            "model_name": "flux-schnell-gguf",
            "beam_endpoint_name": "flux-schnell-gguf",
            "error": str(e)
        }