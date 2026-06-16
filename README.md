# Endpoints Serverless GPU en Beam Cloud

Este directorio contiene las definiciones de los cuatro endpoints de inferencia desplegables en **Beam Cloud**. Cada endpoint ejecuta un modelo de inteligencia artificial específico utilizando aceleración por GPU.

## Modelos y Archivos de los Endpoints

1. **`beam_z_image_turbo.py`** (Modelo: `Tongyi-MAI/Z-Image-Turbo`)
   - Generación rápida basada en SDXL (8 pasos).
   - GPU recomendada: `A10G`.
   - Comando de despliegue:
     ```bash
     beam deploy beam_z_image_turbo.py:generate --name beam-z-image-turbo
     ```

2. **`beam_qwen_image_2512.py`** (Modelo: `Qwen/Qwen-Image-2512`)
   - Generación premium de alta fidelidad, realismo de textos e imágenes complejas.
   - GPU recomendada: `L40S` o `A100`.
   - Comando de despliegue:
     ```bash
     beam deploy beam_qwen_image_2512.py:generate --name beam-qwen-image-2512
     ```

3. **`beam_real_esrgan_x4plus.py`** (Modelo: `RealESRGAN_x4plus`)
   - Reescalado super-resolución de imágenes 4x.
   - GPU recomendada: `T4`.
   - Comando de despliegue:
     ```bash
     beam deploy beam_real_esrgan_x4plus.py:upscale --name beam-real-esrgan-x4plus
     ```

4. **`beam_hat_real_hat_gan_srx4_sharper.py`** (Modelo: `Real_HAT_GAN_SRx4_sharper.pth`)
   - Reescalado premium ultra nítido basado en Hybrid Attention Transformers (HAT).
   - GPU recomendada: `A10G`.
   - Comando de despliegue:
     ```bash
     beam deploy beam_hat_real_hat_gan_srx4_sharper.py:upscale --name beam-hat-real-hat-gan-srx4-sharper
     ```

---

## Instrucciones de Despliegue

1. Instalar la CLI de Beam en tu máquina:
   ```bash
   pip install beam-client
   ```
2. Autenticarte con tu cuenta de Beam:
   ```bash
   beam configure
   ```
3. Ejecutar el comando de despliegue correspondiente dentro de esta carpeta.
4. Anotar las URLs REST API resultantes y el API token de tu cuenta para configurarlas como variables de entorno en la aplicación de Coolify.
