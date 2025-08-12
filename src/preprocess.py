import torch
from PIL import Image
import numpy as np
import os
from transformers import Dinov2Processor, Dinov2Model, CLIPProcessor, CLIPModel

def pil_to_tensor(images_pil, device, dtype):
    if not isinstance(images_pil, list):
        images_pil = [images_pil]
    tensors = []
    for img in images_pil:
        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        tensor = torch.ToTensor()(img) # [C, H, W] in [0, 1]
        tensor = tensor * 2 - 1 # Normalize to [-1, 1]
        tensors.append(tensor)
    return torch.stack(tensors).to(device=device, dtype=dtype)

def tensor_to_pil(images_tensor):
    # images_tensor is [B, C, H, W] in [-1, 1], convert to [0, 255] and PIL
    images_tensor = (images_tensor / 2 + 0.5).clamp(0, 1)
    return [Image.fromarray((255. * img.permute(1,2,0).cpu().numpy()).astype(np.uint8)) for img in images_tensor]

def load_reference_features(config, device, is_testing=False):
    # In a real scenario, this might involve loading a pre-computed feature dataset
    # For this experiment, we generate dummy images and extract features
    print("  Loading/generating dummy natural images for DINOv2 reference...")
    img_size = config['image_size'] if 'image_size' in config else 512 # Default
    if is_testing:
        num_images = 10
        img_size = 64
    else:
        num_images = 500

    dummy_natural_images_pil = []
    for _ in range(num_images):
        # Create a blank image with random color and then add some random noise
        img_array = np.random.randint(0, 256, size=(img_size, img_size, 3), dtype=np.uint8)
        dummy_natural_images_pil.append(Image.fromarray(img_array))

    # Initialize Dinov2 models and processors within this function for feature extraction
    dinov2_model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device).eval()
    dinov2_processor = Dinov2Processor.from_pretrained("facebook/dinov2-base")

    with torch.no_grad():
        ref_inputs = dinov2_processor(images=dummy_natural_images_pil, return_tensors="pt", padding=True).to(device)
        reference_dinov2_features = dinov2_model(pixel_values=ref_inputs.pixel_values).last_hidden_state.mean(dim=1)
    
    print(f"  Loaded {reference_dinov2_features.shape[0]} reference DINOv2 features.")
    return reference_dinov2_features

def precompute_text_embeddings(config, clip_processor, clip_model, device):
    with torch.no_grad():
        prompt_inputs = clip_processor(text=config['prompt'], return_tensors="pt").to(device)
        prompt_embedding = clip_model.get_text_features(input_ids=prompt_inputs.input_ids)
        prompt_embedding = prompt_embedding / prompt_embedding.norm(p=2, dim=-1, keepdim=True)
    return prompt_embedding
