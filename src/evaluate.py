import torch
from diffusers import StableDiffusionPipeline
from torchvision.utils import save_image
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

def generate_and_save_images(
    pipeline, prompts, output_dir, num_inference_steps=50, guidance_scale=7.5, 
    num_images_per_prompt=1, device="cuda"
):
    print(f"Generating images for evaluation in {output_dir}...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_generated_images = []
    for i, prompt in enumerate(prompts):
        with torch.no_grad():
            images = pipeline(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt
            ).images
        
        for j, img in enumerate(images):
            filepath = os.path.join(output_dir, f"generated_image_prompt_{i+1}_sample_{j+1}.png")
            img.save(filepath)
            all_generated_images.append(img)
    print(f"Generated {len(all_generated_images)} images.")
    return all_generated_images

def plot_dummy_metrics(metrics_history, plot_dir="plots"):
    """
    Plots a dummy metric over "epochs" or "checkpoints".
    In a real scenario, this would plot FID, KID, CLIP scores, etc.
    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    print(f"Generating dummy evaluation plots in {plot_dir}...")
    
    steps = list(range(len(metrics_history["g_loss"])))
    
    if metrics_history:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if "g_loss" in metrics_history:
            ax.plot(steps, metrics_history["g_loss"], label='Generator Loss (Evaluation)', marker='o', linestyle='-')
        
        if "hrpm_reward" in metrics_history:
            ax.plot(steps, metrics_history["hrpm_reward"], label='HRPM Reward (Evaluation)', marker='x', linestyle='--')
        
        if "automated_reward" in metrics_history:
            ax.plot(steps, metrics_history["automated_reward"], label='Automated Reward (Evaluation)', marker='s', linestyle=':')

        ax.set_title('Simulated Evaluation Metrics Over Training Steps')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, "simulated_evaluation_metrics.pdf")
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"Saved dummy evaluation plot: {plot_path}")
        plt.close(fig) # Close the plot to free memory
    else:
        print("No evaluation metrics provided to plot.")


if __name__ == "__main__":
    print("Running a dummy evaluation test...")
    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

    test_prompts = ["a cat astronaut floating in space", "a vibrant coral reef"]
    output_dir = ".research/iteration1/images/eval_test"
    
    print("Dummy evaluation image generation completed.")

    dummy_metrics = {
        "g_loss": [1.5, 1.4, 1.3, 1.2, 1.1],
        "hrpm_reward": [0.1, 0.3, 0.5, 0.7, 0.9],
        "automated_reward": [0.2, 0.25, 0.3, 0.35, 0.4]
    }
    plot_dummy_metrics(dummy_metrics, plot_dir="plots/eval_test")
    print("Dummy evaluation plotting completed.")
