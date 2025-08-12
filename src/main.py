import torch
import torch.nn as nn
from accelerate import Accelerator
import os
import shutil
import yaml
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPProcessor, CLIPModel, Dinov2Model, Dinov2Processor

# Import functions from other scripts
from src.preprocess import load_reference_features, precompute_text_embeddings
from src.train import train_model
from src.evaluate import evaluate_model

# Define the path for the config file
CONFIG_PATH = "config/config.yaml"

def main(is_testing=False):
    print(f"\n--- Starting Main Experiment Orchestration (is_testing: {is_testing}) ---")

    # Load configuration
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Override config for testing
    if is_testing:
        print("  Overriding config for testing...")
        config['num_training_steps'] = 2
        config['batch_size'] = 1
        config['num_inference_steps'] = 5
        config['plot_dir'] = "test_plots"
        config['generated_images_dir'] = "test_generated_images"
        # Clean up old test directories
        if os.path.exists(config["plot_dir"]):
            shutil.rmtree(config["plot_dir"])
        if os.path.exists(config["generated_images_dir"]):
            shutil.rmtree(config["generated_images_dir"])
        os.makedirs(config["plot_dir"], exist_ok=True)
        os.makedirs(config["generated_images_dir"], exist_ok=True)


    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    # --- Load Main Diffusion Pipeline ---
    if is_testing:
        print("  Loading minimal models for testing...")
        # Mock StableDiffusionPipeline and its components for testing
        class MockScheduler:
            def __init__(self):
                self.config = {'num_train_timesteps': 1000}
                self.timesteps = torch.linspace(0, 999, config['num_inference_steps']).long().flip(0)
            def set_timesteps(self, num_inference_steps):
                self.timesteps = torch.linspace(0, 999, num_inference_steps).long().flip(0)
            def scale_model_input(self, latents, t): return latents
            def pred_original_sample(self, latents, t, noise_pred): return torch.randn_like(latents) # dummy pred_xstart
            def step(self, noise_pred, t, latents):
                class MockStepOutput:
                    def __init__(self, prev_sample): self.prev_sample = prev_sample
                return MockStepOutput(torch.randn_like(latents)) # dummy prev_sample

        class MockUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type('Config', (object,), {'in_channels': 4, 'sample_size': 64})()
                self.dummy_param = nn.Parameter(torch.zeros(1)) # needs at least one param for optimizer
            def forward(self, x, t, encoder_hidden_states):
                class MockNoisePred:
                    def __init__(self, sample): self.sample = sample
                return MockNoisePred(torch.randn(x.shape[0], x.shape[1], 64, 64, device=x.device, dtype=x.dtype))

        class MockTextEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy_param = nn.Parameter(torch.zeros(1))
            def forward(self, input_ids): return type('Output', (object,), {'last_hidden_state': torch.randn(input_ids.shape[0], 77, 768)})()
        
        class MockPipeline:
            def __init__(self, device, dtype, unet_sample_size):
                self.unet = MockUNet().to(device)
                self.vae = MockUNet().to(device) 
                self.text_encoder = MockTextEncoder().to(device)
                self.scheduler = MockScheduler()
                self.dtype = dtype
                self.unet.config.sample_size = unet_sample_size # Set sample size for mock UNet
            
            def _encode_prompt(self, prompt_text, device, num_images_per_prompt, text_encoder, dtype):
                return torch.randn(num_images_per_prompt * 2, 77, 768, device=device, dtype=dtype)
        
        pipe = MockPipeline(device, torch.float16, 64) # Hardcode sample_size for mock
        config['image_size'] = 64 # Match image size for preprocessing
        
    else:
        print("  Loading full diffusion pipeline...")
        pipe = StableDiffusionPipeline.from_pretrained(config['model_id'], torch_dtype=torch.float16, revision="fp16") # Use fp16 revision if available
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        pipe.vae.eval().requires_grad_(False)
        pipe.text_encoder.eval().requires_grad_(False)
        pipe.unet.train()
        config['image_size'] = pipe.unet.config.sample_size # Get actual image size

    # Prepare for acceleration (only pipe.unet needs to be prepared for gradient tracking)
    # The optimizer will be created and prepared inside train_model.
    # We need to pass a dummy optimizer here for accelerator.prepare(model, optimizer)
    # Or, we just prepare the model. Let's adjust this to just prepare the model.
    # pipe.unet = accelerator.prepare(pipe.unet)
    # Note: For typical Accelerate usage, prepare `model` and `optimizer` together.
    # Since the optimizer is defined within train_model, we just call prepare here to handle `pipe.unet` for device placement/etc.
    # A more robust solution might pass the accelerator to train_model to handle its own prepare() for its optimizer.
    # For this structure, we pass a dummy optimizer to prepare. This is a common pattern for pre-loading models.
    optimizer_dummy = torch.optim.AdamW(pipe.unet.parameters(), lr=config['learning_rate'])
    pipe.unet, optimizer_dummy = accelerator.prepare(pipe.unet, optimizer_dummy) # dummy optimizer just for setup

    # --- Preprocessing Steps ---
    # DINOv2 model and processor are initialized within load_reference_features
    reference_dinov2_features = load_reference_features(config, device, is_testing)

    # Initialize CLIP for text embedding (used by preprocess and train/evaluate)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    prompt_embedding = precompute_text_embeddings(config, clip_processor, clip_model, device)
    
    # --- Training ---
    # The train_model function will internally handle accelerator.prepare for its own optimizer
    trained_pipe = train_model(config, accelerator, pipe, prompt_embedding, reference_dinov2_features, is_testing)

    # --- Evaluation ---
    evaluate_model(config, accelerator, trained_pipe, prompt_embedding, is_testing)

    print("\n--- Main Experiment Orchestration Complete ---")

def test_full_experiment():
    print("\n--- Running Quick Test for Full Experiment Workflow ---")
    try:
        main(is_testing=True)
        
        # Verify plots were attempted to be saved (check directory existence and content)
        test_config_path = "config/config.yaml"
        with open(test_config_path, 'r') as f:
            test_config = yaml.safe_load(f)
        
        assert os.path.exists(test_config["plot_dir"]), "Plot directory was not created."
        # At least one plot file should be created from train.py and one from evaluate.py
        plot_files = [f for f in os.listdir(test_config["plot_dir"]) if f.endswith(".pdf")]
        assert len(plot_files) >= 5, f"Expected at least 5 plot files, found {len(plot_files)}."
        print(f"Successfully generated {len(plot_files)} plot files in '{test_config['plot_dir']}'.")

        assert os.path.exists(test_config["generated_images_dir"]), "Generated images directory was not created."
        generated_image_files = [f for f in os.listdir(test_config["generated_images_dir"]) if f.endswith(".pdf")]
        assert len(generated_image_files) > 0, "No generated image files were created."
        print(f"Successfully generated {len(generated_image_files)} image files in '{test_config['generated_images_dir']}'.")

        print("Quick test passed: Full experiment workflow executed without critical errors and outputs were generated.")
    except Exception as e:
        print(f"Quick test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_full_experiment()

    print("\n--- Running Main Experiment (Conceptual Full Run) ---")
    print("Note: This will attempt to download large models and run for a significant duration.")
    print("For actual full-scale experiments, ensure appropriate hardware and dataset setup.")
    # Uncomment the line below for a full, non-test run
    # main(is_testing=False)
