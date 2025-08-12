import torch
from torchvision import transforms
from PIL import Image
import os
import random

def get_image_transforms(target_size=512, normalize_mean=[0.5], normalize_std=[0.5]):
    """
    Returns image transformation pipeline.
    """
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])

def load_dummy_real_images(num_images=10, image_size=(512, 512), output_dir="data"):
    """
    Generates and saves dummy real images for conceptual discriminator training.
    In a real scenario, this would load actual images from a dataset.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Generating {num_images} dummy real images in {output_dir}...")
    dummy_images = []
    for i in range(num_images):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img = Image.new('RGB', image_size, color=color)
        img_path = os.path.join(output_dir, f"dummy_real_image_{i:03d}.png")
        img.save(img_path)
        dummy_images.append(img_path)
    print("Dummy images generated.")
    return dummy_images

def load_real_images_from_disk(image_paths, transform):
    """
    Loads real images from specified paths and applies transformations.
    """
    print(f"Loading {len(image_paths)} real images from disk...")
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(transform(img))
        except Exception as e:
            print(f"Warning: Could not load image {path} - {e}")
    if not images:
        raise ValueError("No images loaded. Check paths and permissions.")
    return torch.stack(images)

if __name__ == "__main__":
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    dummy_paths = load_dummy_real_images(num_images=20, output_dir=data_dir)
    
    transform = get_image_transforms()
