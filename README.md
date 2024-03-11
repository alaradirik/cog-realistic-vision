# Cog wrapper for Realistic Vision v6.0
This is a cog wrapper for Realistic Vision v6.0 for photorealistic image generation. See the original [repo](https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE), [Civit AI model page](https://civitai.com/models/4201/realistic-vision-v60-b1) and Replicate [demo](https://replicate.com/adirik/realistic-vision-v6.0) for details.


## Basic Usage
You will need to have [Cog](https://github.com/replicate/cog/blob/main/docs/getting-started-own-model.md) and Docker installed to serve your model as an API. Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of the model to [Replicate](https://replicate.com) with Cog. To run a prediction:

```bash
cog predict -i prompt="An astronaut riding a rainbow unicorn"
```

To start your server and serve the model as an API:
```bash
cog run -p 5000 python -m cog.server.http
```

The API input arguments are as follows:

- **prompt:** The input prompt is a text description that guides the image generation process. It should be a detailed and specific description of the desired output image.  
- **negative_prompt:** This parameter allows specifying negative prompts. Negative prompts are terms or descriptions that should be avoided in the generated image, helping to steer the output away from unwanted elements.  
- **width:** This parameter sets the width of the output image.  
- **height:** This parameter sets the height of the output image.  
- **num_outputs:** Specifies the number of images to be generated for a given prompt. This allows for multiple variations of images based on the same input parameters.  
- **scheduler:** The scheduler parameter determines the algorithm used for image generation. Different schedulers can affect the quality and characteristics of the output.  
- **num_steps:** This parameter defines the number of denoising steps in the image generation process.  
- **guidance_scale:** The guidance scale parameter adjusts the influence of the classifier-free guidance in the generation process. Higher values will make the model focus more on the prompt.  
- **seed:** The seed parameter sets a random seed for image generation. A specific seed can be used to reproduce results, or left blank for random generation.   


## Model Details

**Original Model:** https://civitai.com/models/4201/realistic-vision-v60-b1

Some important usage tips from the original model page:

- Best performance comes with the scheduler “DPM++ SDE Karras” which is the default value in the API.  
- Recommended number of denoising steps are: 10+ with DPM++ SDE Karras scheduler / 20+ with DPM++ 2M SDE scheduler.