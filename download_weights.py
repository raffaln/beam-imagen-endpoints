"""
Script to download all model weights to the shared Beam Volume.
Running this via 'beam run' bypasses API gateway timeouts.
"""

import os
import urllib.request
from beam import function, Image as BeamImage, Volume

CACHE_PATH = "./weights"

@function(
    name="beam-weights-downloader",
    gpu="A10G",
    cpu=2,
    memory="16Gi",
    volumes=[Volume(name="weights", mount_path=CACHE_PATH)],
    image=BeamImage(
        python_version="python3.10",
        python_packages=[
            "torch==2.1.2",
            "diffusers>=0.27.0",
            "transformers>=4.38.0",
            "accelerate>=0.27.0",
            "pillow",
            "sentencepiece",
            "spandrel",
            "spandrel_extra_arches",
            "numpy"
        ],
    ),
)
def download_all():
    import torch
    print("--- INICIANDO DESCARGA DE PESOS EN VOLUMEN PERSISTENTE ---")
    
    # Asegurar directorio de weights
    os.makedirs(CACHE_PATH, exist_ok=True)
    
    # 1. Descargar Real-ESRGAN x4plus (pth de GitHub)
    print("\n[1/4] Descargando pesos de RealESRGAN_x4plus...")
    esrgan_file = os.path.join(CACHE_PATH, "RealESRGAN_x4plus.pth")
    if not os.path.exists(esrgan_file):
        urllib.request.urlretrieve(
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            esrgan_file
        )
        print("RealESRGAN_x4plus descargado con éxito.")
    else:
        print("RealESRGAN_x4plus ya existe en volumen cache.")

    # 2. Descargar Real-HAT-GAN SRx4 Sharper (pth de Hugging Face)
    print("\n[2/4] Descargando pesos de Real_HAT_GAN_SRx4_sharper...")
    hat_file = os.path.join(CACHE_PATH, "Real_HAT_GAN_SRx4_sharper.pth")
    if not os.path.exists(hat_file):
        urllib.request.urlretrieve(
            "https://huggingface.co/Acly/hat/resolve/main/Real_HAT_GAN_SRx4_sharper.pth",
            hat_file
        )
        print("Real_HAT_GAN_SRx4_sharper descargado con éxito.")
    else:
        print("Real_HAT_GAN_SRx4_sharper ya existe en volumen cache.")

    # 3. Descargar Z-Image-Turbo (Diffusers pipeline)
    print("\n[3/4] Descargando pipeline Z-Image-Turbo desde Hugging Face...")
    from diffusers import DiffusionPipeline
    pipe_z = DiffusionPipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_PATH,
    )
    print("Z-Image-Turbo descargado y cacheado con éxito.")

    # 4. Descargar Qwen-Image-2512 (Diffusers pipeline)
    print("\n[4/4] Descargando pipeline Qwen-Image-2512 desde Hugging Face...")
    pipe_q = DiffusionPipeline.from_pretrained(
        "Qwen/Qwen-Image-2512",
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_PATH,
    )
    print("Qwen-Image-2512 descargado y cacheado con éxito.")

    print("\n--- TODOS LOS PESOS FUERON DESCARGADOS Y CACHEADOS CON ÉXITO ---")
    return {"status": "success", "message": "All weights downloaded successfully to persistent volume."}
