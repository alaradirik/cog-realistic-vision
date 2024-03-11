# Prediction interface for Cog ⚙️
import os
import math
import subprocess
from typing import List

import torch
from diffusers import (
    StableDiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
    DPMSolverSinglestepScheduler,
    AutoencoderKL,
)
from cog import BasePredictor, Input, Path


MODEL_URL = "https://weights.replicate.delivery/default/realistic-vision/realistic-vision-v6.tar"
VAE_NAME = "stabilityai/sd-vae-ft-mse-original"
VAE_CKPT = "vae-ft-mse-840000-ema-pruned.ckpt"
MODEL_CACHE = "cache"
VAE_CACHE = "vae-cache"


def download_weights(url, dest):
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)


class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


# for this model, recommended scheduler is DPM++ SDE Karras
class DPMppSDEKarras:
    def from_config(config):
        return DPMSolverSinglestepScheduler.from_config(config, use_karras_sigmas=True)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "DPM++_SDE_Karras": DPMppSDEKarras,
}


class Predictor(BasePredictor):
    def base(self, x):
        return int(8 * math.floor(int(x) / 8))

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        vae = AutoencoderKL.from_single_file(
            "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors",
            cache_dir=VAE_CACHE,
        )
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        self.pipe = StableDiffusionPipeline.from_pretrained(MODEL_CACHE, vae=vae)
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe = self.pipe.to("cuda")

    def predict(
        self,
        prompt: str = "RAW photo, a portrait photo of a latina woman in casual clothes, natural skin, 8k uhd, high quality, film grain, Fujifilm XT3",
        negative_prompt: str = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
        num_steps: int = Input(
            description="Number of diffusion steps", ge=0, le=100, default=20
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=2
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        scheduler: str = Input(
            description="Scheduler to use, DPM++ SDE Karras is recommended",
            choices=SCHEDULERS.keys(),
            default="DPM++_SDE_Karras",
        ),
        width: int = Input(description="Width", ge=0, le=1920, default=512),
        height: int = Input(description="Height", ge=0, le=1920, default=728),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), byteorder="big")
        print(f"Using seed: {seed}")

        generator = torch.Generator("cuda").manual_seed(seed)

        width = self.base(width)
        height = self.base(height)

        self.pipe.scheduler = SCHEDULERS[scheduler].from_config(
            self.pipe.scheduler.config
        )

        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_steps,
            "width": width,
            "height": height,
        }

        output = self.pipe(**common_args)

        output_paths = []
        for i, image in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
