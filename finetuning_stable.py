import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
import bitsandbytes as bnb
import utils

device =  "cuda" if torch.cuda.is_available() else "cpu"
#preparing const and get device
SD_FINETUNED_PATH = "./cache/stable-diffusion-v2-1-finetuned"
SD_LOCAL_PATH = "./cache/stable-diffusion-v2-1-local"
SD_REMOTE_PATH = "stabilityai/stable-diffusion-2-1"
def get_sd_local_or_download():
    if os.path.isdir(SD_LOCAL_PATH):
        print("Use local model")
        pipe = StableDiffusionPipeline.from_pretrained(SD_LOCAL_PATH)
    else:
        print("Download model")
        pipe = StableDiffusionPipeline.from_pretrained(
            SD_REMOTE_PATH,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe.save_pretrained(SD_LOCAL_PATH)
    return pipe

def finetuning_sd_model(
    object_images, # tensor type
    object_masks, # image type
    object_promts,
    num_epochs = 50,
    lr = 1e-5,
    seed: int = 42
):
    losses = []

    scaler = GradScaler()
    generator = torch.Generator(device=device).manual_seed(seed)
    if (os.path.isdir(SD_LOCAL_PATH)):
        image_pipe = StableDiffusionPipeline.from_pretrained(
            SD_LOCAL_PATH,
            safety_checker=None,
            requires_safety_checker=False
        )
    else:
      image_pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        safety_checker=None,
        requires_safety_checker=False
      )
      image_pipe.save_pretrained(SD_LOCAL_PATH)
    image_pipe.enable_xformers_memory_efficient_attention()
    optimizer = bnb.optim.AdamW8bit(image_pipe.unet.parameters(), lr=lr)
    image_pipe.scheduler = DDIMScheduler.from_config(image_pipe.scheduler.config)
    image_pipe.enable_model_cpu_offload()
    vae = image_pipe.vae
    tokenizer = image_pipe.tokenizer
    text_encoder = image_pipe.text_encoder
    unet = image_pipe.unet
    scheduler = image_pipe.scheduler
    text_encoder.to(device)
    unet.to(device)

    # create mask preprocessing by vae_scale_factor
    mask_preprocess = transforms.Compose(
    [
        transforms.Resize((512 // image_pipe.vae_scale_factor, 512 // image_pipe.vae_scale_factor)),
        transforms.ToTensor()
    ])

    # preprocess masks and encode objects image.
    with torch.no_grad():
      z_masks = []
      for mask in object_masks:
        z_masks.append(mask_preprocess(mask).to(device))
      z_images = []
      for image in object_images:
        z_images.append(vae.encode(image.unsqueeze(0)).latent_dist.sample(generator=generator))

    # function to encode prompt.
    def get_text_embeddings(prompt):
      text_inputs = tokenizer(
          prompt, padding="max_length", max_length=tokenizer.model_max_length,
          truncation=True, return_tensors="pt"
      )
      text_input_ids = text_inputs.input_ids.to(device)
      uncond_inputs = tokenizer(
          "", padding="max_length", max_length=text_input_ids.shape[-1],
          return_tensors="pt"
      ).input_ids.to(device)
      with torch.no_grad():
          prompt_embeds = text_encoder(text_input_ids)[0]
          uncond_embeds = text_encoder(uncond_inputs)[0]
      return torch.cat([uncond_embeds, prompt_embeds])
    

    # encode prompt.
    with torch.no_grad():
      z_prompts = []
      for prompt in object_promts:
        z_prompts.append(get_text_embeddings(prompt))

    for epoch in range(num_epochs):
        for z_image, z_mask, z_prompt in tqdm(zip(z_images, z_masks, z_prompts), desc=f"Epoch {epoch + 1}/{num_epochs}"):
          noise = torch.randn(
            size = (1, unet.config.in_channels, 512 // image_pipe.vae_scale_factor, 512 // image_pipe.vae_scale_factor),
            generator=generator,
            device=device,
            dtype=unet.dtype)
          timesteps = torch.randint(
            low = 0,
            hight = image_pipe.scheduler.num_train_timesteps,
            size = (1,),
            device=device,
            ).long()
          
          # create noise and add to latent of the image.
          noise = torch.randn_like(z_image)
          noise_image = image_pipe.scheduler.add_noise(z_image, noise, timesteps)
          latent_model_input = torch.cat([noise_image] * 2)
          latent_model_input = scheduler.scale_model_input(latent_model_input, timesteps)

          #create predict and calculate loss
          with autocast(dtype=torch.float16):
            noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states=z_prompt).sample
            loss = F.mse_loss(
              noise_pred * z_mask, noise_image*z_mask
            )  # NB - trying to predict noise (eps) not (noisy_ims-clean_ims) or just (clean_ims)

          # Store for later plotting
          losses.append(loss.item())

          # Update the model parameters with the optimizer based on this loss
          scaler.scale(loss).backward()
          scaler.step(optimizer)
          scaler.update()
          optimizer.zero_grad()

    # Save finetuned model and plot the losses.
    image_pipe.save_pretrained(SD_FINETUNED_PATH, safe_serialization=True)
    print("Model saved")
    plt.plot(losses)

@torch.no_grad()
def scene_level_construction(
    grouped_gen_object,
    grouped_gen_mask,
    bg_prompt,
    gl_prompt,
    alpha: float = 0.8,
    num_inference_steps: int = 50,
    device = "gpu",
    guidance_scale: float = 7.5,
    seed: int = 42
):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    if (os.path.isdir(SD_FINETUNED_PATH)):
        image_pipe = StableDiffusionPipeline.from_pretrained(
            SD_FINETUNED_PATH,
            safety_checker=None,
            requires_safety_checker=False
        )
    else:
      image_pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        safety_checker=None,
        requires_safety_checker=False
      )
      image_pipe.save_pretrained(SD_LOCAL_PATH)
    image_pipe.scheduler = DDIMScheduler.from_config(image_pipe.scheduler.config)
    image_pipe.enable_model_cpu_offload()
    vae = image_pipe.vae
    tokenizer = image_pipe.tokenizer
    text_encoder = image_pipe.text_encoder
    unet = image_pipe.unet
    scheduler = image_pipe.scheduler
    text_encoder.to(device)
    unet.to(device)

    def get_text_embeddings(prompt):
        text_inputs = tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(device)
        uncond_inputs = tokenizer(
            "", padding="max_length", max_length=text_input_ids.shape[-1],
            return_tensors="pt"
        ).input_ids.to(device)

        with torch.no_grad():
            prompt_embeds = text_encoder(text_input_ids)[0]
            uncond_embeds = text_encoder(uncond_inputs)[0]

        return torch.cat([uncond_embeds, prompt_embeds])

    bg_embeds = get_text_embeddings(bg_prompt)
    global_embeds = get_text_embeddings(gl_prompt)

    # 1. Get initial foreground latent z_init
    z_init = vae.encode(grouped_gen_object.unsqueeze(0)).latent_dist.sample(generator=generator)
    mask_preprocess = transforms.Compose(
    [
        transforms.Resize((512 // image_pipe.vae_scale_factor, 512 // image_pipe.vae_scale_factor)),
        transforms.ToTensor()
    ])
    z_mask = mask_preprocess(grouped_gen_mask).to(device)
    z_init = z_init * vae.config.scaling_factor
    # 4. Initialize latents
    latents = torch.randn(
        (1, unet.config.in_channels, 512 // image_pipe.vae_scale_factor, 512 // image_pipe.vae_scale_factor),
        generator=generator,
        device=device,
        dtype=unet.dtype
    )

    # 5. Set timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # 6. Denoising Loop
    transition_step = int(num_inference_steps * alpha)

    for i, t in enumerate(tqdm(timesteps, desc="Diffusion Denoising")):
        # Expand latents for guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # Main algorithm logic
        if i < transition_step:
            # Step 5: denoise(z_t, P_b, t) -> to get the background
            with torch.no_grad():
                noise_pred_bg = unet(latent_model_input, t, encoder_hidden_states=bg_embeds).sample

            # Perform guidance
            noise_pred_uncond_bg, noise_pred_text_bg = noise_pred_bg.chunk(2)
            noise_pred_bg = noise_pred_uncond_bg + guidance_scale * (noise_pred_text_bg - noise_pred_uncond_bg)

            # Get z_bg by taking one step with the scheduler
            z_bg = scheduler.step(noise_pred_bg, t, latents).prev_sample

            # Step 6: noise(z_init, t) -> to get the foreground structure
            noise = torch.randn_like(z_init, device=device)
            z_fg = scheduler.add_noise(z_init, noise, t)
            # Step 7: Combine z_bg and z_fg using the mask
            latents = z_bg * (1 - z_mask) + z_fg * z_mask

        else:
            # Step 9: denoise(z_t, P_g, t) -> harmonizing step
            with torch.no_grad():
                noise_pred_global = unet(latent_model_input, t, encoder_hidden_states=global_embeds).sample

            # Perform guidance
            noise_pred_uncond_global, noise_pred_text_global = noise_pred_global.chunk(2)
            noise_pred_global = noise_pred_uncond_global + guidance_scale * (noise_pred_text_global - noise_pred_uncond_global)

            # Update latents with one step
            latents = scheduler.step(noise_pred_global, t, latents).prev_sample
    # 12. Decode the final latent z_0 to get the image x_hat
    latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()

    #save image to drive
    image = Image.fromarray((image[0] * 255).astype(np.uint8))
    utils.saveImage(image, folder = "scene_guide_result", prefix = "scene")
