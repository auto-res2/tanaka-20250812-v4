import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import StableDiffusionPipeline, DDPOScheduler
from accelerate import Accelerator
from transformers import CLIPVisionModel, CLIPImageProcessor 
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml

# Import custom modules
from src.models import HumanPreferenceRewardModel, Discriminator
from src.preprocess import get_image_transforms, load_dummy_real_images, load_real_images_from_disk
from src.train import train_hrpm_step, train_discriminator_step, train_policy_step
from src.evaluate import generate_and_save_images, plot_dummy_metrics

# --- 0. Configuration Loading ---
def load_config(config_path="config/experiment_config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# --- Test Function ---
def run_quick_test(config, device):
    print("\n--- Running quick test to verify functionality ---")
    
    # Test HRPM
    try:
        hrpm_test = HumanPreferenceRewardModel(clip_model_name=config["hrpm"]["clip_model_name"]).to(device)
        dummy_image = Image.new('RGB', (224, 224), color='blue') # CLIP expects 224x224 or resizes
        score = hrpm_test([dummy_image])
        print(f"HRPM test successful. Dummy score: {score.item():.4f}")
    except Exception as e:
        print(f"HRPM test failed: {e}")
        return False

    # Test Discriminator
    try:
        discriminator_test = Discriminator(img_channels=config["discriminator"]["img_channels"],
                                         features_d=config["discriminator"]["features_d"]).to(device)
        dummy_input = torch.randn(1, 3, 512, 512, device=device) # SDXL output size
        output = discriminator_test(dummy_input)
        print(f"Discriminator test successful. Output shape: {output.shape} (should be 1x1x1x1)")
        if output.shape != (1, 1, 1, 1):
             print(f"WARNING: Discriminator output shape is not 1x1x1x1, which is usually expected for GANs. Got {output.shape}")
    except Exception as e:
        print(f"Discriminator test failed: {e}")
        return False

    print("--- Quick test completed successfully! ---")
    return True

# --- Main Training Loop ---
if __name__ == "__main__":
    # Load configuration
    config = load_config()

    accelerator = Accelerator()
    device = accelerator.device
    
    # Set seed for reproducibility
    torch.manual_seed(config["experiment_params"]["seed"])
    np.random.seed(config["experiment_params"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["experiment_params"]["seed"])

    # Run quick test first
    if not run_quick_test(config, device):
        accelerator.print("Exiting due to failed quick test.")
        exit()

    # Create directories
    os.makedirs(config["experiment_params"]["plot_output_dir"], exist_ok=True)
    os.makedirs(config["experiment_params"]["image_output_dir"], exist_ok=True)
    os.makedirs(config["experiment_params"]["model_save_dir"], exist_ok=True)
    os.makedirs("data", exist_ok=True) # Ensure data directory exists

    # Load Models
    accelerator.print("Loading Stable Diffusion Pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(config["experiment_params"]["pretrained_model_name_or_path"],
                                                   torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    unet = pipe.unet
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    scheduler = DDPOScheduler(pipe.scheduler.config) # Use DDPOScheduler
    
    accelerator.print("Initializing HRPM and Discriminator...")
    hrpm = HumanPreferenceRewardModel(clip_model_name=config["hrpm"]["clip_model_name"]).to(device)
    discriminator = Discriminator(img_channels=config["discriminator"]["img_channels"],
                                 features_d=config["discriminator"]["features_d"]).to(device)

    # Optimizers
    optim_g = optim.AdamW(unet.parameters(), lr=config["generator"]["learning_rate"])
    optim_d = optim.AdamW(discriminator.parameters(), lr=config["discriminator"]["learning_rate"])
    optim_hrpm = optim.AdamW(hrpm.preference_head.parameters(), lr=config["hrpm"]["learning_rate"])

    # Prepare models and optimizers for distributed training
    unet, optim_g, hrpm, discriminator, optim_d, optim_hrpm = accelerator.prepare(
        unet, optim_g, hrpm, discriminator, optim_d, optim_hrpm
    )

    # Dummy data/prompts for illustration
    prompts = [
        "a photo of a majestic cat sitting on a throne", 
        "a highly detailed painting of a futuristic cyberpunk city at night with neon lights", 
        "a close-up of a vibrant sunflower in full bloom under a clear sky",
        "a serene landscape with a calm lake and distant mountains, hyperrealistic",
        "a whimsical illustration of a flying teapot in a starry sky",
        "an intricate robotic hand holding a delicate glass sphere, studio lighting"
    ]

    # Prepare real images for discriminator training
    dummy_real_image_paths = load_dummy_real_images(
        num_images=20, image_size=(512, 512), output_dir="data"
    )
    real_images_transform = get_image_transforms()
    real_image_tensors_for_discriminator = load_real_images_from_disk(dummy_real_image_paths, real_images_transform).to(device)


    num_training_steps = config["experiment_params"]["num_training_steps"]
    hrpm_update_interval = config["experiment_params"]["hrpm_update_interval"]
    evaluation_interval = config["experiment_params"]["evaluation_interval"]
    lambda_gan_value = config["experiment_params"]["lambda_gan"]

    bce_loss = nn.BCEWithLogitsLoss()

    # Store metrics for plotting
    all_experiments_metrics = {}

    for exp_name, exp_config in config["experiments"].items():
        accelerator.print(f"\n--- Starting training for: {exp_name} ---")
        
        # Re-initialize models and optimizers for each experiment to ensure fair comparison
        # (This is a simplification for a single script. In a real setup, you'd run separate jobs)
        unet = accelerator.unwrap_model(unet)
        hrpm = accelerator.unwrap_model(hrpm)
        discriminator = accelerator.unwrap_model(discriminator)

        pipe_fresh = StableDiffusionPipeline.from_pretrained(config["experiment_params"]["pretrained_model_name_or_path"],
                                                             torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        unet = pipe_fresh.unet
        vae = pipe_fresh.vae
        tokenizer = pipe_fresh.tokenizer
        text_encoder = pipe_fresh.text_encoder
        scheduler = DDPOScheduler(pipe_fresh.scheduler.config)

        hrpm = HumanPreferenceRewardModel(clip_model_name=config["hrpm"]["clip_model_name"]).to(device)
        discriminator = Discriminator(img_channels=config["discriminator"]["img_channels"],
                                     features_d=config["discriminator"]["features_d"]).to(device)

        optim_g = optim.AdamW(unet.parameters(), lr=config["generator"]["learning_rate"])
        optim_d = optim.AdamW(discriminator.parameters(), lr=config["discriminator"]["learning_rate"])
        optim_hrpm = optim.AdamW(hrpm.preference_head.parameters(), lr=config["hrpm"]["learning_rate"])
        
        unet, optim_g, hrpm, discriminator, optim_d, optim_hrpm = accelerator.prepare(
            unet, optim_g, hrpm, discriminator, optim_d, optim_hrpm
        )

        current_g_losses = []
        current_hrpm_rewards = []
        current_automated_rewards = []
        current_d_losses = []
        current_gan_g_losses = []

        for step in range(num_training_steps):
            # Sample prompts for this step
            sample_prompts = np.random.choice(prompts, size=4, replace=False).tolist()
            
            # --- Discriminator Training (only if adversarial component is enabled) ---
            d_loss = 0.0 # Default if GAN is not used
            if exp_config["use_gan_regularization"]:
                with torch.no_grad():
                    dummy_latents = torch.randn(4, unet.config.in_channels,
                                                unet.config.sample_size, unet.config.sample_size,
                                                device=device,
                                                generator=torch.Generator(device=device).manual_seed(np.random.randint(100000)))
                    dummy_images_unnorm = vae.decode(dummy_latents / vae.config.scaling_factor, return_dict=False)[0]
                    dummy_generated_images_for_d = (dummy_images_unnorm / 2 + 0.5).clamp(0, 1) # Normalize to [0,1]
                
                sample_real_images = real_image_tensors_for_discriminator[torch.randperm(real_image_tensors_for_discriminator.size(0))[:4]]

                d_loss = train_discriminator_step(
                    discriminator, optim_d, 
                    sample_real_images, 
                    dummy_generated_images_for_d, bce_loss
                )
            
            # --- Policy Optimization (Generator) ---
            g_loss, avg_hrpm_reward, avg_automated_reward, avg_gan_loss = train_policy_step(
                accelerator, unet, vae, tokenizer, text_encoder, scheduler,
                hrpm, discriminator, optim_g, sample_prompts,
                lambda_gan=lambda_gan_value,
                use_hrpm=exp_config["use_hrpm"],
                use_gan_regularization=exp_config["use_gan_regularization"]
            )

            # Store metrics
            current_g_losses.append(g_loss)
            current_hrpm_rewards.append(avg_hrpm_reward)
            current_automated_rewards.append(avg_automated_reward)
            current_d_losses.append(d_loss)
            current_gan_g_losses.append(avg_gan_loss)

            accelerator.print(f"[{exp_name}] Step {step+1}/{num_training_steps}: G Loss={g_loss:.4f}, HRPM Reward={avg_hrpm_reward:.4f}, Auto Reward={avg_automated_reward:.4f}, D Loss={d_loss:.4f}, GAN Loss_G={avg_gan_loss:.4f}")

            # --- Periodic HRPM Refinement (only for HRPM-enabled groups) ---
            if exp_config["use_hrpm"] and (step + 1) % hrpm_update_interval == 0:
                accelerator.print(f"  [{exp_name}] Collecting new human preference data and refining HRPM...")
                with torch.no_grad():
                    eval_pipe = StableDiffusionPipeline(
                        vae=accelerator.unwrap_model(vae),
                        text_encoder=accelerator.unwrap_model(text_encoder),
                        tokenizer=accelerator.unwrap_model(tokenizer),
                        unet=accelerator.unwrap_model(unet),
                        scheduler=scheduler,
                        safety_checker=None, feature_extractor=None, requires_safety_checker=False
                    )
                    eval_pipe.to(device)
                    hrpm_eval_prompts = np.random.choice(prompts, size=2, replace=False).tolist()
                    hrpm_gen_images = eval_pipe(hrpm_eval_prompts, num_inference_steps=20).images
                
                if len(hrpm_gen_images) >= 2:
                    dummy_preference_data = [(hrpm_gen_images[0], hrpm_gen_images[1], 1)] # Image 0 preferred over Image 1
                    for _ in range(2): # Mini HRPM update loop (more iterations in real training)
                        hrpm_loss = train_hrpm_step(hrpm, optim_hrpm, dummy_preference_data, device)
                        accelerator.print(f"    [{exp_name}] HRPM refinement loss: {hrpm_loss:.4f}")
                else:
                    accelerator.print(f"    [{exp_name}] Not enough images generated for HRPM refinement.")


            # --- Periodic Evaluation and Image Generation ---
            if (step + 1) % evaluation_interval == 0 or step == num_training_steps - 1:
                accelerator.print(f"  [{exp_name}] Simulating evaluation at step {step+1}...")
                
                eval_pipe = StableDiffusionPipeline(
                    vae=accelerator.unwrap_model(vae),
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    tokenizer=accelerator.unwrap_model(tokenizer),
                    unet=accelerator.unwrap_model(unet),
                    scheduler=scheduler,
                    safety_checker=None, feature_extractor=None, requires_safety_checker=False
                )
                eval_pipe.to(device)
                
                eval_output_dir = os.path.join(config["experiment_params"]["image_output_dir"], exp_name.replace(" ", "_"), f"step_{step+1}")
                generated_images_for_eval = generate_and_save_images(
                    eval_pipe, 
                    prompts[:2],
                    eval_output_dir,
                    num_inference_steps=25
                )

        # Store final metrics for the experiment group
        all_experiments_metrics[exp_name] = {
            "g_loss": current_g_losses,
            "hrpm_reward": current_hrpm_rewards,
            "automated_reward": current_automated_rewards,
            "d_loss": current_d_losses,
            "gan_loss_g": current_gan_g_losses
        }

    accelerator.print("\nAll training simulations complete. Generating plots...")

    # --- Plotting Results ---
    plot_dir = config["experiment_params"]["plot_output_dir"]
    os.makedirs(plot_dir, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Training Metrics Across Experimental Groups', fontsize=16)
    
    # Plot 1: Generator Loss
    ax = axs[0, 0]
    for exp_name, data in all_experiments_metrics.items():
        if data["g_loss"]:
            ax.plot(data["g_loss"], label=exp_name)
    ax.set_title('Generator (UNet) Loss')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)

    # Plot 2: Reward (HRPM vs Automated)
    ax = axs[0, 1]
    for exp_name, data in all_experiments_metrics.items():
        if "HRPM" in exp_name and data["hrpm_reward"]:
            ax.plot(data["hrpm_reward"], label=f'{exp_name} (HRPM Reward)', linestyle='-')
        elif data["automated_reward"] and not config["experiments"][exp_name]["use_hrpm"]:
            ax.plot(data["automated_reward"], label=f'{exp_name} (Automated Reward)', linestyle='--')
        elif exp_name == "DDPO Baseline":
            ax.plot(data["automated_reward"], label=f'{exp_name} (Automated Reward)', linestyle='--')
            
    ax.set_title('Average Reward (HRPM or Automated)')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Reward Score')
    ax.legend()
    ax.grid(True)

    # Plot 3: Discriminator Loss
    ax = axs[1, 0]
    for exp_name, data in all_experiments_metrics.items():
        if config["experiments"][exp_name]["use_gan_regularization"] and data["d_loss"]:
            ax.plot(data["d_loss"], label=exp_name, alpha=0.7)
    ax.set_title('Discriminator Loss')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    if any(cfg["use_gan_regularization"] for cfg in config["experiments"].values()):
        ax.legend()
    ax.grid(True)

    # Plot 4: Generator's GAN Loss Component
    ax = axs[1, 1]
    for exp_name, data in all_experiments_metrics.items():
        if config["experiments"][exp_name]["use_gan_regularization"] and data["gan_loss_g"]:
            ax.plot(data["gan_loss_g"], label=exp_name, alpha=0.7)
    ax.set_title('Generator GAN Loss Component')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss (from Discriminator)')
    if any(cfg["use_gan_regularization"] for cfg in config["experiments"].values()):
        ax.legend()
    ax.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(plot_dir, "training_metrics_comparison.pdf"), bbox_inches="tight")
    accelerator.print(f"Plot saved: {os.path.join(plot_dir, 'training_metrics_comparison.pdf')}")

    # Individual plots for specific conditions
    if "Full Proposed Framework" in all_experiments_metrics and all_experiments_metrics["Full Proposed Framework"]["g_loss"]:
        plt.figure(figsize=(8, 6))
        plt.plot(all_experiments_metrics["Full Proposed Framework"]["g_loss"], label='Full Framework Generator Loss')
        plt.title('Full Proposed Framework: Generator Loss')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(plot_dir, "training_loss_full_framework.pdf"), bbox_inches="tight")
        accelerator.print(f"Plot saved: {os.path.join(plot_dir, 'training_loss_full_framework.pdf')}")

    if "Full Proposed Framework" in all_experiments_metrics and "DDPO + HRPM" in all_experiments_metrics and \
       all_experiments_metrics["Full Proposed Framework"]["hrpm_reward"] and all_experiments_metrics["DDPO + HRPM"]["hrpm_reward"]:
        plt.figure(figsize=(8, 6))
        plt.plot(all_experiments_metrics["Full Proposed Framework"]["hrpm_reward"], label='Full Framework (HRPM Reward)', color='green')
        plt.plot(all_experiments_metrics["DDPO + HRPM"]["hrpm_reward"], label='DDPO + HRPM (HRPM Reward)', color='purple', linestyle='--')
        plt.title('Human Preference Reward Comparison')
        plt.xlabel('Training Step')
        plt.ylabel('HRPM Score')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(plot_dir, "hrpm_reward_comparison.pdf"), bbox_inches="tight")
        accelerator.print(f"Plot saved: {os.path.join(plot_dir, 'hrpm_reward_comparison.pdf')}")

    accelerator.print("Experiment simulation complete and plots generated.")
