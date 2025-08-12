import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPProcessor, CLIPModel, Dinov2Model, Dinov2Processor
from PIL import Image
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# Import helper functions from preprocess
from src.preprocess import pil_to_tensor, tensor_to_pil

# --- Dummy Aesthetic Predictor ---
class AestheticPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # CLIP feature dim is 768 for ViT-L/14
        self.linear = nn.Linear(768, 1) 
        # Initialize with random weights for dummy purposes
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.uniform_(-0.5, 0.5)

    def forward(self, clip_features):
        return self.linear(clip_features)

# --- Helper Functions for Regularization Losses ---
def compute_vlm_loss_fn(images_pil, prompt_embedding_ref, clip_processor, clip_model, device):
    if not images_pil:
        return torch.zeros(1, device=device) # Return 0 if no images
    inputs = clip_processor(images=images_pil, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        image_features = clip_model.get_image_features(pixel_values=inputs.pixel_values)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    
    # Ensure prompt_embedding_ref is expanded correctly for batch comparison
    # It should be [1, D] and expand to [B, D]
    expanded_prompt_embedding = prompt_embedding_ref.expand_as(image_features)
    
    loss = 1.0 - F.cosine_similarity(image_features, expanded_prompt_embedding, dim=-1) # Penalize low similarity
    return loss

def compute_dist_loss_fn(images_pil, reference_features, dinov2_processor, dinov2_model, device):
    if not images_pil:
        return torch.zeros(1, device=device) # Return 0 if no images
    inputs = dinov2_processor(images=images_pil, return_tensors="pt", padding=True).to(device)
    with torch.no_grad(): # DINOv2 model is frozen
        # The .last_hidden_state gives features for patches, mean(dim=1) aggregates
        image_features = dinov2_model(pixel_values=inputs.pixel_values).last_hidden_state.mean(dim=1)
    
    if reference_features.shape[0] == 0: # Handle empty reference features for test cases
        return torch.zeros(image_features.shape[0], device=device) # Return 0s if no reference features

    # Use a subset of reference features if the full set is too large
    num_samples_ref = min(image_features.shape[0] * 8, reference_features.shape[0]) # Heuristic for sample size
    sampled_ref_indices = random.sample(range(reference_features.shape[0]), num_samples_ref)
    sampled_ref_features = reference_features[sampled_ref_indices]

    # Compute pairwise L2 distances and take minimum for each generated image
    dist_matrix = torch.cdist(image_features, sampled_ref_features, p=2)
    min_distances = dist_matrix.min(dim=1)[0]
    return min_distances

def compute_consistency_loss_fn(image_t_pil, image_t_minus_1_pil, clip_processor, clip_model, device):
    if not image_t_pil or not image_t_minus_1_pil:
        return torch.zeros(1, device=device) # Return 0 if no images or previous images
    
    inputs_t = clip_processor(images=image_t_pil, return_tensors="pt", padding=True).to(device)
    inputs_t_minus_1 = clip_processor(images=image_t_minus_1_pil, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad(): # CLIP model is frozen
        features_t = clip_model.get_image_features(pixel_values=inputs_t.pixel_values)
        features_t_minus_1 = clip_model.get_image_features(pixel_values=inputs_t_minus_1.pixel_values)
    
    features_t = features_t / features_t.norm(p=2, dim=-1, keepdim=True)
    features_t_minus_1 = features_t_minus_1 / features_t_minus_1.norm(p=2, dim=-1, keepdim=True)
    
    # Using 1 - cosine_similarity for semantic consistency
    loss = 1.0 - F.cosine_similarity(features_t, features_t_minus_1, dim=-1) # Higher values mean less consistent
    return loss

def train_model(config, accelerator, pipe, prompt_embedding, reference_dinov2_features, is_testing=False):
    # Unpack config
    model_id = config['model_id']
    prompt = config['prompt']
    num_training_steps = config['num_training_steps']
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    lambda_vlm = config['lambda_vlm']
    lambda_dist = config['lambda_dist']
    lambda_consistency = config['lambda_consistency']
    num_inference_steps = config['num_inference_steps']
    guidance_scale = config['guidance_scale']
    seed = config['seed']
    consistency_check_interval = config['consistency_check_interval']
    plot_dir = config['plot_dir']
    generated_images_dir = config['generated_images_dir']

    print(f"\n--- Running Training (is_testing: {is_testing}) ---")
    
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(generated_images_dir, exist_ok=True)

    device = accelerator.device

    # --- Load Auxiliary Models ---
    # These are loaded here because they are used within the training loop
    if is_testing:
        print("  Loading minimal auxiliary models for testing...")
        # Mock CLIP, DINOv2, Aesthetic Predictor
        clip_model_vlm = AestheticPredictor().to(device).eval() 
        clip_processor_vlm = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14") 
        dinov2_model = AestheticPredictor().to(device).eval() 
        dinov2_processor = Dinov2Processor.from_pretrained("facebook/dinov2-base")
        aesthetic_predictor = AestheticPredictor().to(device).eval()
    else:
        print("  Loading full auxiliary models for experiment...")
        clip_model_vlm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        clip_processor_vlm = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        dinov2_model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device).eval()
        dinov2_processor = Dinov2Processor.from_pretrained("facebook/dinov2-base")
        aesthetic_predictor = AestheticPredictor().to(device).eval()

    optimizer = optim.AdamW(pipe.unet.parameters(), lr=learning_rate)
    pipe.unet, optimizer = accelerator.prepare(pipe.unet, optimizer)

    # --- Training Loop ---
    # Store metrics for plotting
    train_losses = []
    train_rewards = []
    train_vlm_losses = []
    train_dist_losses = []
    train_consistency_losses = []

    print(f"Starting training for {num_training_steps} steps...")
    
    # Use tqdm for a progress bar
    for step in tqdm(range(num_training_steps), desc="Training"):
        pipe.unet.train()
        
        current_latents = torch.randn(
            (batch_size, pipe.unet.config.in_channels, pipe.unet.config.sample_size, pipe.unet.config.sample_size),
            generator=torch.Generator(device=device).manual_seed(seed + step + accelerator.process_index),
            device=device,
            dtype=pipe.unet.dtype
        )
        
        text_embeddings_cond = pipe._encode_prompt(prompt, device, batch_size, pipe.text_encoder, pipe.unet.dtype)
        text_embeddings_uncond = pipe._encode_prompt("", device, batch_size, pipe.text_encoder, pipe.unet.dtype)
        text_embeddings = torch.cat([text_embeddings_uncond, text_embeddings_cond])

        pipe.scheduler.set_timesteps(num_inference_steps)
        timesteps = pipe.scheduler.timesteps
        
        per_trajectory_vlm_losses_step = torch.zeros(batch_size, device=device)
        per_trajectory_dist_losses_step = torch.zeros(batch_size, device=device)
        per_trajectory_consistency_losses_step = torch.zeros(batch_size, device=device)

        prev_pred_xstart_images_pil = [None] * batch_size

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([current_latents] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Get pred_xstart (x_0 prediction from current step's noise prediction)
            pred_xstart = pipe.scheduler.pred_original_sample(current_latents, t, noise_pred)
            
            # Convert pred_xstart to PIL for VLM/DINO
            pred_xstart_images_pil = tensor_to_pil(pred_xstart)
            
            # Compute regularization losses for this step
            per_trajectory_vlm_losses_step += compute_vlm_loss_fn(pred_xstart_images_pil, prompt_embedding, clip_processor_vlm, clip_model_vlm, device)
            per_trajectory_dist_losses_step += compute_dist_loss_fn(pred_xstart_images_pil, reference_dinov2_features, dinov2_processor, dinov2_model, device)

            if i > 0 and (i + 1) % consistency_check_interval == 0:
                for b in range(batch_size):
                    # Ensure prev_pred_xstart_images_pil[b] is not None for consistency check
                    if prev_pred_xstart_images_pil[b] is not None:
                        per_trajectory_consistency_losses_step[b] += compute_consistency_loss_fn(
                            [pred_xstart_images_pil[b]], # Current image (list for batching)
                            [prev_pred_xstart_images_pil[b]], # Previous image (list for batching)
                            clip_processor_vlm, clip_model_vlm, device
                        )
            
            prev_pred_xstart_images_pil = pred_xstart_images_pil # Update for next step's consistency check

            # Apply noise prediction to get latents for the next step
            current_latents = pipe.scheduler.step(noise_pred, t, current_latents).prev_sample

        # After completing the trajectory for the batch
        final_images_pil = tensor_to_pil(current_latents)

        # 2. Calculate Primary Reward (R_t for the final image)
        final_image_inputs_clip = clip_processor_vlm(images=final_images_pil, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            final_image_features_clip = clip_model_vlm.get_image_features(pixel_values=final_image_inputs_clip.pixel_values)
            final_image_features_clip = final_image_features_clip / final_image_features_clip.norm(p=2, dim=-1, keepdim=True)
            
            clip_score = F.cosine_similarity(final_image_features_clip, prompt_embedding.expand_as(final_image_features_clip), dim=-1)
            aesthetic_score = aesthetic_predictor(final_image_features_clip).squeeze()
        
        # Composite primary reward (example: normalize scores if needed, then average)
        primary_rewards = (clip_score * 0.7 + (aesthetic_score.sigmoid() * 5.0) * 0.3) # Sigmoid aesthetic score to [0, 5] then scale
        
        # 3. Policy Gradient Update (J_regularized)
        # Normalize regularization losses by number of steps they were computed over
        num_vlm_dist_steps = num_inference_steps
        num_consistency_steps = num_inference_steps // consistency_check_interval
        
        # Ensure division by non-zero
        avg_vlm_loss = per_trajectory_vlm_losses_step / num_vlm_dist_steps if num_vlm_dist_steps > 0 else per_trajectory_vlm_losses_step
        avg_dist_loss = per_trajectory_dist_losses_step / num_vlm_dist_steps if num_vlm_dist_steps > 0 else per_trajectory_dist_losses_step
        avg_consistency_loss = per_trajectory_consistency_losses_step / num_consistency_steps if num_consistency_steps > 0 else per_trajectory_consistency_losses_step

        policy_loss_per_trajectory = (
            -primary_rewards 
            + lambda_vlm * avg_vlm_loss 
            + lambda_dist * avg_dist_loss 
            + lambda_consistency * avg_consistency_loss
        )
        
        total_policy_loss = policy_loss_per_trajectory.mean()

        # Backpropagation and Optimization
        accelerator.backward(total_policy_loss)
        
        if (step + 1) % gradient_accumulation_steps == 0:
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(pipe.unet.parameters(), 1.0) # Gradient clipping
            optimizer.step()
            optimizer.zero_grad()
            
        # Store metrics for plotting (after each effective update)
        if (step + 1) % gradient_accumulation_steps == 0:
            train_losses.append(total_policy_loss.item())
            train_rewards.append(primary_rewards.mean().item())
            train_vlm_losses.append(avg_vlm_loss.mean().item())
            train_dist_losses.append(avg_dist_loss.mean().item())
            train_consistency_losses.append(avg_consistency_loss.mean().item())

        # --- Logging and Monitoring ---
        if accelerator.is_main_process and (step + 1) % 50 == 0:
            print(f"\nStep {step+1}/{num_training_steps}:")
            print(f"  Primary Reward: {primary_rewards.mean().item():.4f}")
            print(f"  Avg VLM Reg Loss (per trajectory): {avg_vlm_loss.mean().item():.4f}")
            print(f"  Avg Dist Reg Loss (per trajectory): {avg_dist_loss.mean().item():.4f}")
            print(f"  Avg Consistency Reg Loss (per trajectory): {avg_consistency_loss.mean().item():.4f}")
            print(f"  Total Policy Loss: {total_policy_loss.item():.4f}")

            # Save a sample image for qualitative assessment
            if not is_testing:
                sample_image_path = os.path.join(generated_images_dir, f"step_{step+1}.pdf")
                # For demonstration, we save the first image in the batch
                final_images_pil[0].save(sample_image_path)
                print(f"  Sample image saved to {sample_image_path}")
            else:
                print("  Skipping image save in test mode.")

    accelerator.wait_for_everyone()
    # Save the model (example)
    if accelerator.is_main_process and not is_testing:
        unwrapped_unet = accelerator.unwrap_model(pipe.unet)
        model_save_path = os.path.join("models", model_id.replace("/", "_"), "unet_finetuned_regularized")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        unwrapped_unet.save_pretrained(model_save_path)
        print(f"Finetuned UNet saved to {model_save_path}.")

    print("Training complete.")
    
    # --- Plotting Training Progress ---
    if accelerator.is_main_process and len(train_losses) > 0:
        print("Generating training plots...")
        steps_per_update = gradient_accumulation_steps
        plot_steps = [i * steps_per_update for i in range(len(train_losses))]

        plt.figure(figsize=(10, 6))
        sns.lineplot(x=plot_steps, y=train_losses)
        plt.title('Training Policy Loss Over Steps (Regularized AMICT)')
        plt.xlabel('Training Steps')
        plt.ylabel('Total Policy Loss')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "training_loss_amict.pdf"), bbox_inches="tight")
        plt.close()
        print(f"Saved {os.path.join(plot_dir, 'training_loss_amict.pdf')}")

        plt.figure(figsize=(10, 6))
        sns.lineplot(x=plot_steps, y=train_rewards)
        plt.title('Primary Reward Over Steps (Regularized AMICT)')
        plt.xlabel('Training Steps')
        plt.ylabel('Primary Reward')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "primary_reward_amict.pdf"), bbox_inches="tight")
        plt.close()
        print(f"Saved {os.path.join(plot_dir, 'primary_reward_amict.pdf')}")

        plt.figure(figsize=(10, 6))
        sns.lineplot(x=plot_steps, y=train_vlm_losses, label='L_VLM')
        sns.lineplot(x=plot_steps, y=train_dist_losses, label='L_dist')
        sns.lineplot(x=plot_steps, y=train_consistency_losses, label='L_consistency')
        plt.title('Regularization Losses Over Steps (Regularized AMICT)')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "regularization_losses_amict.pdf"), bbox_inches="tight")
        plt.close()
        print(f"Saved {os.path.join(plot_dir, 'regularization_losses_amict.pdf')}")

    return pipe # Return the trained pipe
